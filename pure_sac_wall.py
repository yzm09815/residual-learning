import os
import numpy as np
import warnings
import gymnasium as gym_new
import gym
import time
import signal

warnings.filterwarnings("ignore")

from gymnasium.wrappers import TimeLimit
from gymnasium.spaces.utils import flatdim, flatten
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs import test_cases as tc
from gym_collision_avoidance.envs.policies.ExternalPolicy import ExternalPolicy

# ==========================================
# 1. ORCA 行人动作计算（温和版）
# ==========================================
def calculate_orca_pedestrian_action(host_agent, agents,
                                     tau=3.0,              # ORCA 时间视野（越大越早避让）
                                     max_speed_ratio=0.7,  # 行人最大速度 = pref_speed * ratio
                                     responsiveness=0.5):  # 响应灵敏度 [0,1]，越低越"懒"
    """
    简化 ORCA 行人模型：
    - 比 APF 更温和：不会产生巨大斥力把自己弹飞
    - 模拟真实行人：慢速、惯性大、反应迟钝
    - 核心思路：计算 ORCA 半平面约束，在可行速度集合中选最接近期望速度的
    """
    pos = host_agent.pos_global_frame
    vel = host_agent.vel_global_frame if hasattr(host_agent, 'vel_global_frame') else np.zeros(2)
    goal = host_agent.goal_global_frame
    radius = host_agent.radius
    pref_speed = host_agent.pref_speed * max_speed_ratio  # 行人走得慢

    # 期望速度：朝目标方向，速度为 pref_speed
    goal_vec = goal - pos
    dist_goal = np.linalg.norm(goal_vec)
    if dist_goal > 0.1:
        v_pref = (goal_vec / dist_goal) * pref_speed
    else:
        return np.array([0.0, 0.0])  # 到目标了，停下

    # 收集所有 ORCA 约束（半平面）
    orca_lines = []  # 每个元素: (point_on_line, direction_of_line)

    for other in agents:
        if other.id == host_agent.id:
            continue
        
        other_pos = other.pos_global_frame
        other_vel = other.vel_global_frame if hasattr(other, 'vel_global_frame') else np.zeros(2)
        other_radius = other.radius

        # 相对位置和速度
        rel_pos = other_pos - pos
        rel_vel = vel - other_vel
        dist = np.linalg.norm(rel_pos)
        combined_radius = radius + other_radius

        # 如果已经重叠，直接推开
        if dist < combined_radius:
            if dist > 0.001:
                push_dir = -rel_pos / dist
            else:
                push_dir = np.array([1.0, 0.0])
            orca_lines.append((np.zeros(2), push_dir))
            continue

        # ORCA 速度障碍物计算
        # 截断锥体（truncated VO）
        inv_tau = 1.0 / tau
        
        # 相对速度在 VO 锥体中的位置
        w = rel_vel - inv_tau * rel_pos  # 相对于截断圆心的向量
        w_len = np.linalg.norm(w)
        
        if w_len < 0.0001:
            # 相对速度几乎为零，用位置方向
            w_unit = -rel_pos / dist if dist > 0.001 else np.array([1.0, 0.0])
        else:
            w_unit = w / w_len
        
        # 法线方向（指向 VO 外部）
        # 简化：用 w 的方向作为约束法线
        line_dir = np.array([-w_unit[1], w_unit[0]])  # 旋转 90°
        
        # 约束点：当前速度需要调整的最小量
        u = (combined_radius * inv_tau - w_len) * w_unit
        
        # 只承担一半的避让责任（ORCA 的核心：双方各让一半）
        # 行人版：自己承担更少（responsiveness < 0.5 意味着更"懒"）
        orca_point = vel + responsiveness * u
        
        orca_lines.append((orca_point, line_dir))

    # 在 ORCA 约束下找最优速度
    # 简化方法：从 v_pref 出发，逐步投影到可行区域
    v_opt = v_pref.copy()
    
    for point, direction in orca_lines:
        # 检查 v_opt 是否违反此约束
        det = np.cross(direction, v_opt - point)
        if det < 0:
            # 违反约束，投影到约束线上
            proj = np.dot(v_opt - point, direction)
            v_opt = point + proj * direction

    # 限速（行人不会跑太快）
    speed = np.linalg.norm(v_opt)
    if speed > pref_speed:
        v_opt = v_opt / speed * pref_speed

    # 转换为 [speed, heading] 格式
    final_speed = np.linalg.norm(v_opt)
    final_heading = np.arctan2(v_opt[1], v_opt[0])
    
    return np.array([final_speed, final_heading])


# ==========================================
# 0. 超时保护
# ==========================================
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("操作超时！")

# ==========================================
# 0. 万能转接头
# ==========================================
class UniversalGymAdapter(gym_new.Wrapper):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        self.__dict__['metadata'] = getattr(env, 'metadata', {'render_modes': []})
        self.__dict__['_metadata'] = self.__dict__['metadata']
        self.__dict__['render_mode'] = getattr(env, 'render_mode', None)
        
        try:
            self.spec = getattr(env, 'spec', None)
        except AttributeError:
            pass
        
        self.reset_timeout_count = 0
        self.max_reset_retries = 5
    
    def reset(self, **kwargs):
        for attempt in range(self.max_reset_retries):
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(3)
                ret = self.env.reset()
                signal.alarm(0)
                if isinstance(ret, tuple) and len(ret) == 2:
                    return ret
                return ret, {}
            except TimeoutError:
                signal.alarm(0)
                self.reset_timeout_count += 1
                if attempt < self.max_reset_retries - 1:
                    print(f"⚠️  Reset 超时，重试 {attempt+1}/{self.max_reset_retries}...")
                    continue
                else:
                    raise RuntimeError(f"Reset 连续 {self.max_reset_retries} 次超时，放弃")
    
    def step(self, action):
        ret = self.env.step(action)
        if len(ret) == 4:
            obs, reward, done, info = ret
            return obs, reward, done, False, info
        elif len(ret) == 5:
            return ret
        else:
            raise ValueError(f"环境返回了 {len(ret)} 个值，无法处理！")
            
    def render(self):
        return self.env.render()


# ==========================================
# 1. 单智能体包装器（v6：其他智能体用 ORCA 行人策略）
#    v7: 仅新增墙壁碰撞检测（不改 APF/ORCA/观测/奖励）
# ==========================================
WALL_HALF_SIZE = 6.0  # ★★★ v7 新增：边界墙位置，匹配 side_length=12 ★★★

class SimpleSingleAgentWrapper(gym_new.Wrapper):
    """
    v6 核心改动：
    - Agent 0 = SAC 控制（不变）
    - Agent 1/2/3 = ORCA 温和行人（替代 APF 机器人）
      · 速度更慢（pref_speed × 0.7）
      · 反应更迟钝（responsiveness=0.5）
      · 不会产生巨大斥力弹飞自己
      · 更接近真实行人的避障行为

    v7: 仅新增撞墙判定 -> done + reward=-10
    """
    def __init__(self, env):
        super().__init__(env)
        
        self.action_space = gym_new.spaces.Box(
            low=np.array([0.0, -np.pi]),
            high=np.array([2.0, np.pi]),
            dtype=np.float32
        )
        
        self.last_dist = None
        self.episode_count = 0
        self.step_count = 0
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_dist = None
        self.step_count = 0
        
        if hasattr(self.env.unwrapped, 'agents'):
            agents = self.env.unwrapped.agents
            agents[0].is_learning = True
            for i in range(1, len(agents)):
                agents[i].is_learning = False
        
        return obs
    
    def _check_collision_agent0(self, agent0, agents):
        for other in agents:
            if other.id == agent0.id:
                continue
            dist = np.linalg.norm(agent0.pos_global_frame - other.pos_global_frame)
            if dist < (agent0.radius + other.radius):
                return True
        return False
    
    def _check_wall_collision(self, agent0):
        """★★★ v7 新增：检测是否出界（碰到边界墙）★★★"""
        x, y = agent0.pos_global_frame
        r = agent0.radius
        s = WALL_HALF_SIZE
        return (x - r < -s or x + r > s or y - r < -s or y + r > s)
    
    def _get_min_dist_to_others(self, agent0, agents):
        min_dist_edge = float('inf')
        for other in agents:
            if other.id == agent0.id:
                continue
            dist = np.linalg.norm(agent0.pos_global_frame - other.pos_global_frame)
            dist_edge = dist - agent0.radius - other.radius
            if dist_edge < min_dist_edge:
                min_dist_edge = dist_edge
        return min_dist_edge
    
    def step(self, action_agent0):
        raw_env = self.env.unwrapped
        agents = raw_env.agents

        # 清除 Agent 0 的冻结状态
        agents[0].ran_out_of_time = False
        agents[0].time_remaining_to_reach_goal = 999.0
        agents[0].is_at_goal = False
        agents[0].was_at_goal_already = False
        agents[0].is_done = False

        # 扩展 Agent 0 的 history 数组
        agent0 = agents[0]
        if hasattr(agent0, 'step_num') and hasattr(agent0, 'global_state_history'):
            if agent0.step_num >= agent0.global_state_history.shape[0] - 1:
                for attr_name in dir(agent0):
                    if attr_name.endswith('_history') and not attr_name.startswith('_'):
                        try:
                            history = getattr(agent0, attr_name)
                            if isinstance(history, np.ndarray) and len(history.shape) >= 1:
                                new_size = history.shape[0] * 2
                                if len(history.shape) == 2:
                                    new_history = np.zeros((new_size, history.shape[1]))
                                    new_history[:history.shape[0], :] = history
                                elif len(history.shape) == 1:
                                    new_history = np.zeros(new_size)
                                    new_history[:history.shape[0]] = history
                                else:
                                    continue
                                setattr(agent0, attr_name, new_history)
                        except:
                            pass

        # 为所有智能体生成动作
        all_actions = []
        for i, agent in enumerate(agents):
            if i == 0:
                # Agent 0: 纯 SAC 控制
                all_actions.append(action_agent0)
            else:
                # ========== v6 核心改动：使用 ORCA 行人策略 ==========
                action_orca = calculate_orca_pedestrian_action(
                    agent, agents,
                    tau=3.0,              # 时间视野（行人看远一点，提前让路）
                    max_speed_ratio=0.7,  # 行人走路速度 = 70% pref_speed
                    responsiveness=0.5    # 只承担一半避让责任（像真人一样"懒"）
                )
                all_actions.append(action_orca)
        
        obs, reward_env, done_env, truncated, info = self.env.step(np.array(all_actions))
        self.step_count += 1

        if isinstance(reward_env, (list, tuple, np.ndarray)):
            reward_env = float(reward_env[0])
        
        # 自主判断 done
        agent0 = raw_env.agents[0]
        dist_to_goal = np.linalg.norm(
            agent0.goal_global_frame - agent0.pos_global_frame
        )

        done = False
        done_reason = None
        
        if dist_to_goal < 0.5:
            done = True
            done_reason = "reached_goal"
        
        if not done:
            collision_detected = False
            if hasattr(agent0, 'in_collision') and agent0.in_collision:
                collision_detected = True
            elif hasattr(agent0, 'is_in_collision') and agent0.is_in_collision:
                collision_detected = True
            if not collision_detected:
                collision_detected = self._check_collision_agent0(agent0, agents)
            if collision_detected:
                done = True
                done_reason = "collision"
        
        # ★★★ v7 新增：撞墙检测 ★★★
        if not done and self._check_wall_collision(agent0):
            done = True
            done_reason = "wall_collision"
        
        # ==========================================
        # 🔧 v6 "半稀疏奖励"
        # ==========================================
        if done or truncated:
            if done_reason == "reached_goal" or dist_to_goal < 0.5:
                reward = 10.0
                if self.episode_count % 100 == 0:
                    print(f"    ✅ Episode {self.episode_count}: 成功到达目标！"
                          f"(步数={self.step_count}, 奖励=+10.0)")
            elif done_reason == "collision":
                reward = -10.0
                if self.episode_count % 100 == 0:
                    print(f"    ❌ Episode {self.episode_count}: 碰撞！"
                          f"(步数={self.step_count}, 奖励=-10.0)")
            elif done_reason == "wall_collision":
                reward = -10.0
                if self.episode_count % 100 == 0:
                    print(f"    🧱 Episode {self.episode_count}: 撞墙！"
                          f"(步数={self.step_count}, 奖励=-10.0)")
            elif truncated:
                reward = -5.0
                if self.episode_count % 100 == 0:
                    print(f"    ⏰ Episode {self.episode_count}: 超时！"
                          f"(步数={self.step_count}, 距离={dist_to_goal:.2f}, 奖励=-5.0)")
            else:
                reward = -5.0
                if self.episode_count % 100 == 0:
                    print(f"    ⚠️ Episode {self.episode_count}: 其他结束({done_reason})！"
                          f"(步数={self.step_count}, 距离={dist_to_goal:.2f}, 奖励=-5.0)")
            
            self.episode_count += 1
            self.last_dist = None
        else:
            # --- 半稀疏奖励：过程中只有极微弱的引导 ---
            reward = -0.01  # 每步小惩罚（鼓励快速完成）
            
            # 🕯️ "小夜灯"：极其微弱的距离引导（权重 0.2，原来是 2.0）
            if self.last_dist is not None:
                dist_change = self.last_dist - dist_to_goal
                reward += dist_change * 0.7
            
            # 🚨 "近地警报"：靠近行人 < 0.5m 时持续扣分（软性安全边界）
            # 这是解决 reward 震荡和撞车率的关键
            min_dist_edge = self._get_min_dist_to_others(agent0, agents)
            if min_dist_edge < 0.5:
                # 距离越近惩罚越大：0m → -0.5，0.5m → 0
                proximity_penalty = -0.5 * (1.0 - min_dist_edge / 0.5)
                reward += proximity_penalty
            
            # ★★★ v7 新增：近墙警报，距墙 < 0.5m 时扣 -0.1（比行人惩罚弱5倍，避免怕墙）★★★
            x, y = agent0.pos_global_frame
            s = WALL_HALF_SIZE
            min_wall_dist = min(s - abs(x), s - abs(y)) - agent0.radius
            if min_wall_dist < 0.5:
                reward -= 0.1 * (1.0 - min_wall_dist / 0.5)
            
            self.last_dist = dist_to_goal
        
        return obs, reward, done, truncated, info


# ==========================================
# 1.5 混合场景包装器（70% 固定 + 30% 随机）
# ==========================================
class MixedScenarioWrapper(gym_new.Wrapper):
    """
    v6 核心改动：混合场景
    - 70% 概率使用 6 种固定场景（轮换）
    - 30% 概率随机生成场景
    
    随机场景：
    - 4 个智能体随机分布在圆环上
    - 目标点在对侧半圆
    - 保证初始无碰撞
    
    行人参数：
    - Agent 0（SAC 控制）：pref_speed=1.0, radius=0.3（标准机器人）
    - Agent 1/2/3（ORCA 行人）：pref_speed=0.8~1.2（随机），radius=0.25~0.35（随机）
      模拟真实行人的体型和步速差异
    """
    def __init__(self, env, random_ratio=0.5):
        super().__init__(env)
        self.scenario_type = 0
        self.random_ratio = random_ratio  # 随机场景比例
        self.total_resets = 0
        self.random_count = 0
        self.fixed_count = 0
    
    def _generate_random_scenario(self, raw_env):
        """
        随机生成 4 智能体场景
        - 智能体分布在半径 3.5~5.0 的圆环上
        - 角度随机但保证间隔 > 45°（避免初始重叠）
        - 目标在对侧
        """
        num_agents = len(raw_env.agents)
        spawn_radius = np.random.uniform(2.0, 4.0)  # ★★★ v7 修改：原 3.5~5.0 → 2.0~4.0，离墙留 2m 安全距离 ★★★
        
        # 随机生成 4 个角度（保证最小间隔 > 45°）
        angles = []
        for _ in range(100):  # 最多尝试 100 次
            angles = sorted(np.random.uniform(0, 2 * np.pi, num_agents))
            min_gap = min(
                min(angles[i+1] - angles[i] for i in range(len(angles)-1)),
                (2 * np.pi - angles[-1] + angles[0])
            )
            if min_gap > np.pi / 4:  # 45° 最小间隔
                break
        else:
            # 兜底：均匀分布 + 小随机扰动
            angles = [i * 2 * np.pi / num_agents + np.random.uniform(-0.3, 0.3)
                      for i in range(num_agents)]
        
        for i, agent in enumerate(raw_env.agents):
            # 起点
            angle = angles[i]
            x = spawn_radius * np.cos(angle)
            y = spawn_radius * np.sin(angle)
            agent.pos_global_frame = np.array([x, y])
            
            # 目标：对侧（加随机偏移）
            goal_angle = angle + np.pi + np.random.uniform(-0.5, 0.5)
            goal_radius = np.random.uniform(2.0, 4.0)  # ★★★ v7 修改：原 3.0~5.0 → 2.0~4.0，避免目标贴墙 ★★★
            gx = goal_radius * np.cos(goal_angle)
            gy = goal_radius * np.sin(goal_angle)
            agent.goal_global_frame = np.array([gx, gy])
            
            # 朝向目标
            goal_vec = agent.goal_global_frame - agent.pos_global_frame
            agent.heading_global_frame = np.arctan2(goal_vec[1], goal_vec[0])
            
            agent.vel_global_frame = np.array([0.0, 0.0])
        
        # 设置行人参数
        self._set_pedestrian_params(raw_env)
    
    def _set_pedestrian_params(self, raw_env):
        """
        设置智能体参数：
        - Agent 0：标准机器人参数
        - Agent 1/2/3：模拟行人（速度和体型有随机波动）
        """
        # Agent 0：SAC 控制的机器人
        raw_env.agents[0].pref_speed = 1.0
        raw_env.agents[0].radius = 0.3
        
        # Agent 1/2/3：ORCA 行人（更温和）
        for i in range(1, len(raw_env.agents)):
            # 行人速度：0.8~1.2 m/s（正常步行速度范围）
            raw_env.agents[i].pref_speed = np.random.uniform(0.8, 1.2)
            # 行人半径：0.25~0.35 m（体型差异）
            raw_env.agents[i].radius = np.random.uniform(0.25, 0.35)
            raw_env.agents[i].vel_global_frame = np.array([0.0, 0.0])
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        raw_env = self.env.unwrapped
        if not hasattr(raw_env, 'agents') or len(raw_env.agents) < 4:
            return obs, info
        
        self.total_resets += 1
        
        # ========== 70% 固定 / 30% 随机 ==========
        if np.random.random() < self.random_ratio:
            # ===== 30%：随机场景 =====
            self._generate_random_scenario(raw_env)
            self.random_count += 1
            if self.total_resets % 200 == 0:
                print(f"    🎲 随机场景 (累计: {self.random_count}/{self.total_resets}, "
                      f"比例={self.random_count/self.total_resets:.1%})")
        else:
            # ===== 70%：固定场景（轮换 6 种）=====
            scenario = self.scenario_type % 6
            self._apply_fixed_scenario(raw_env, scenario)
            self._set_pedestrian_params(raw_env)
            self.scenario_type += 1
            self.fixed_count += 1
        
        # 重新计算观测
        try:
            obs = raw_env._get_obs(raw_env.agents)
            if isinstance(obs, dict) and 0 in obs:
                obs = obs[0]
        except:
            pass
        
        return obs, info
    
    def _apply_fixed_scenario(self, raw_env, scenario):
        """6 种固定 4 智能体场景"""
        if scenario == 0:
            # 场景 1：十字交叉
            raw_env.agents[0].pos_global_frame = np.array([-4.0, 0.0])
            raw_env.agents[0].goal_global_frame = np.array([4.0, 0.0])
            raw_env.agents[0].heading_global_frame = 0.0
            
            raw_env.agents[1].pos_global_frame = np.array([4.0, 0.0])
            raw_env.agents[1].goal_global_frame = np.array([-4.0, 0.0])
            raw_env.agents[1].heading_global_frame = np.pi
            
            raw_env.agents[2].pos_global_frame = np.array([0.0, -4.0])
            raw_env.agents[2].goal_global_frame = np.array([0.0, 4.0])
            raw_env.agents[2].heading_global_frame = np.pi / 2
            
            raw_env.agents[3].pos_global_frame = np.array([0.0, 4.0])
            raw_env.agents[3].goal_global_frame = np.array([0.0, -4.0])
            raw_env.agents[3].heading_global_frame = -np.pi / 2
                
        elif scenario == 1:
            # 场景 2：环形交换（对角互换）
            raw_env.agents[0].pos_global_frame = np.array([-3.0, -3.0])
            raw_env.agents[0].goal_global_frame = np.array([3.0, 3.0])
            raw_env.agents[0].heading_global_frame = np.pi / 4
            
            raw_env.agents[1].pos_global_frame = np.array([3.0, -3.0])
            raw_env.agents[1].goal_global_frame = np.array([-3.0, 3.0])
            raw_env.agents[1].heading_global_frame = 3 * np.pi / 4
            
            raw_env.agents[2].pos_global_frame = np.array([-3.0, 3.0])
            raw_env.agents[2].goal_global_frame = np.array([3.0, -3.0])
            raw_env.agents[2].heading_global_frame = -np.pi / 4
            
            raw_env.agents[3].pos_global_frame = np.array([3.0, 3.0])
            raw_env.agents[3].goal_global_frame = np.array([-3.0, -3.0])
            raw_env.agents[3].heading_global_frame = -3 * np.pi / 4
                
        elif scenario == 2:
            # 场景 3：平行对向
            raw_env.agents[0].pos_global_frame = np.array([-4.0, 1.5])
            raw_env.agents[0].goal_global_frame = np.array([4.0, 1.5])
            raw_env.agents[0].heading_global_frame = 0.0
            
            raw_env.agents[1].pos_global_frame = np.array([4.0, 1.5])
            raw_env.agents[1].goal_global_frame = np.array([-4.0, 1.5])
            raw_env.agents[1].heading_global_frame = np.pi
            
            raw_env.agents[2].pos_global_frame = np.array([-4.0, -1.5])
            raw_env.agents[2].goal_global_frame = np.array([4.0, -1.5])
            raw_env.agents[2].heading_global_frame = 0.0
            
            raw_env.agents[3].pos_global_frame = np.array([4.0, -1.5])
            raw_env.agents[3].goal_global_frame = np.array([-4.0, -1.5])
            raw_env.agents[3].heading_global_frame = np.pi
                
        elif scenario == 3:
            # 场景 4：扇形汇聚
            raw_env.agents[0].pos_global_frame = np.array([-4.0, 0.0])
            raw_env.agents[0].goal_global_frame = np.array([4.0, 0.0])
            raw_env.agents[0].heading_global_frame = 0.0
            
            raw_env.agents[1].pos_global_frame = np.array([0.0, 4.0])
            raw_env.agents[1].goal_global_frame = np.array([0.0, -4.0])
            raw_env.agents[1].heading_global_frame = -np.pi / 2
            
            raw_env.agents[2].pos_global_frame = np.array([2.0, 3.5])
            raw_env.agents[2].goal_global_frame = np.array([-2.0, -3.5])
            raw_env.agents[2].heading_global_frame = -2.1
            
            raw_env.agents[3].pos_global_frame = np.array([-1.0, -4.0])
            raw_env.agents[3].goal_global_frame = np.array([1.0, 4.0])
            raw_env.agents[3].heading_global_frame = np.pi / 2
                
        elif scenario == 4:
            # 场景 5：星形散开
            raw_env.agents[0].pos_global_frame = np.array([-1.0, 0.0])
            raw_env.agents[0].goal_global_frame = np.array([4.0, 0.0])
            raw_env.agents[0].heading_global_frame = 0.0
            
            raw_env.agents[1].pos_global_frame = np.array([1.0, 0.0])
            raw_env.agents[1].goal_global_frame = np.array([-4.0, 0.0])
            raw_env.agents[1].heading_global_frame = np.pi
            
            raw_env.agents[2].pos_global_frame = np.array([0.0, -1.0])
            raw_env.agents[2].goal_global_frame = np.array([0.0, 4.0])
            raw_env.agents[2].heading_global_frame = np.pi / 2
            
            raw_env.agents[3].pos_global_frame = np.array([0.0, 1.0])
            raw_env.agents[3].goal_global_frame = np.array([0.0, -4.0])
            raw_env.agents[3].heading_global_frame = -np.pi / 2
                
        else:
            # 场景 6：斜角混战
            raw_env.agents[0].pos_global_frame = np.array([-3.5, 1.0])
            raw_env.agents[0].goal_global_frame = np.array([3.5, -1.0])
            raw_env.agents[0].heading_global_frame = -0.27
            
            raw_env.agents[1].pos_global_frame = np.array([2.0, 3.0])
            raw_env.agents[1].goal_global_frame = np.array([-2.0, -3.0])
            raw_env.agents[1].heading_global_frame = -2.16
            
            raw_env.agents[2].pos_global_frame = np.array([3.5, -2.0])
            raw_env.agents[2].goal_global_frame = np.array([-3.5, 2.0])
            raw_env.agents[2].heading_global_frame = 2.58
            
            raw_env.agents[3].pos_global_frame = np.array([-1.0, -3.5])
            raw_env.agents[3].goal_global_frame = np.array([1.0, 3.5])
            raw_env.agents[3].heading_global_frame = 1.29
        
        for agent in raw_env.agents:
            agent.vel_global_frame = np.array([0.0, 0.0])

# ==========================================
# 3. 观测展平器
# ==========================================
class FlattenObservationWrapper(gym_new.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.is_multi_agent = False
        original_space = env.observation_space
        if isinstance(original_space, gym_new.spaces.Dict):
            if 0 in original_space.spaces:
                print(">>> 检测到多智能体环境，将提取 Agent 0 的观测")
                self.is_multi_agent = True
        
        print(">>> 正在探测实际观测维度...")
        try:
            dummy_obs = env.reset()
            if isinstance(dummy_obs, tuple):
                dummy_obs = dummy_obs[0]
            if self.is_multi_agent and isinstance(dummy_obs, dict) and 0 in dummy_obs:
                dummy_obs = dummy_obs[0]
            flat_dummy = self._recursive_flatten(dummy_obs)
            self.flat_dim = len(flat_dummy)
            print(f">>> ✅ 检测到实际观测维度: {self.flat_dim}")
        except Exception as e:
            print(f"⚠️ 维度探测失败 ({e})，使用默认值 108")
            self.flat_dim = 108
        
        self.observation_space = gym_new.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.flat_dim,), dtype=np.float32
        )
    
    def _recursive_flatten(self, obj):
        if isinstance(obj, dict):
            flat_list = []
            for key in sorted(obj.keys()):
                flat_list.extend(self._recursive_flatten(obj[key]))
            return flat_list
        elif isinstance(obj, (list, tuple)):
            flat_list = []
            for item in obj:
                flat_list.extend(self._recursive_flatten(item))
            return flat_list
        elif isinstance(obj, np.ndarray):
            return obj.flatten().tolist()
        elif isinstance(obj, (int, float, np.integer, np.floating)):
            return [float(obj)]
        else:
            try:
                return [float(obj)]
            except:
                return [0.0]

    def observation(self, observation):
        if self.is_multi_agent and isinstance(observation, dict):
            if 0 in observation:
                observation = observation[0]
            else:
                observation = observation[list(observation.keys())[0]]
        if isinstance(observation, tuple):
            observation = observation[0]
        try:
            flat_list = self._recursive_flatten(observation)
            return np.array(flat_list, dtype=np.float32)
        except Exception as e:
            print(f"❌ 展平失败: {e}")
            return np.zeros(self.flat_dim, dtype=np.float32)


# ==========================================
# 3.5 实时进度回调
# ==========================================
class RealTimeProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, check_freq=1000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.last_time = None
        self.start_time = None
    
    def _on_training_start(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        print("\n🚀 训练开始！")
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            now = time.time()
            elapsed = now - self.last_time
            fps = self.check_freq / elapsed if elapsed > 0 else 0
            progress = self.num_timesteps / self.total_timesteps * 100
            total_elapsed = now - self.start_time
            if self.num_timesteps > 0:
                eta_seconds = (self.total_timesteps - self.num_timesteps) * total_elapsed / self.num_timesteps
                eta_minutes = int(eta_seconds / 60)
                print(f"⏳ {progress:.1f}% | {self.num_timesteps}/{self.total_timesteps} 步 | "
                      f"FPS: {fps:.0f} | 预计剩余: {eta_minutes} 分钟")
            self.last_time = now
        return True


# ==========================================
# 4. 训练主流程
# ==========================================
def make_env():
    env = gym.make("CollisionAvoidance-v0")
    
    env.unwrapped.test_case_args = {
        'num_agents': 4,
        'side_length': 12,
        'policies': 'external'
    }
    
    # 1. 适配器 + 超时保护
    env = UniversalGymAdapter(env)
    
    # 2. 时间限制（400 步）
    env = TimeLimit(env, max_episode_steps=400)
    
    # 3. 单智能体包装器（v7：v6 + 墙壁碰撞检测）
    # 纯 SAC 模式：动作空间就是 [0, 2.0] 和 [-np.pi, np.pi]
    env = SimpleSingleAgentWrapper(env)
    
    # 4. 混合场景 Wrapper（70% 固定 + 30% 随机）
    env = MixedScenarioWrapper(env, random_ratio=0.3)
    
    # 5. [已移除] 残差 APF Wrapper
    
    # 6. 观测展平器
    env = FlattenObservationWrapper(env)
    
    # 7. Monitor
    env = Monitor(env)
    
    return env



if __name__ == "__main__":
    # ★ 更改了 log 保存路径，避免和 residual 混淆 ★
    log_dir = "./sac_pure_logs_v7_walls/"
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 60)
    print("🎯 Pure SAC RL 训练 (纯SAC对照组，无APF残差)")
    print("=" * 60)
    print("\n  除了移除 APF 残差动作逻辑外，完全保持：")
    print("  🧱 边界墙 ±6.0（匹配 side_length=12）")
    print("  🧱 撞墙 -> done + reward=-10（和撞人一样）")
    print("  ✅ ORCA 行人策略 / 观测维度 / 奖励惩罚函数 全部不变")
    print("=" * 60)
    
    print("\n>>> 正在初始化环境...")
    env = DummyVecEnv([make_env])
    
    print("\n>>> 正在构建 SAC 模型...")
    policy_kwargs = dict(
        net_arch=[512, 256, 128]
    )
    
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        buffer_size=300000,
        batch_size=256,
        ent_coef='auto',
        tau=0.005,
        gamma=0.99,
    )
    
    total_steps = 1500000
    
    print(f"\n>>> 开始训练... (总步数: {total_steps})")
    
    progress_callback = RealTimeProgressCallback(
        total_timesteps=total_steps,
        check_freq=5000
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=log_dir,
        name_prefix='sac_pure_v7_walls'  # 改了 checkpoint 前缀
    )
    
    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_callback, progress_callback],
        log_interval=10
    )
    
    model.save("sac_pure_v7_walls_final")
    print("\n>>> ✅ 训练完成！")