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
# 全局训练进度追踪器
# ==========================================
class TrainingProgress:
    """Wrapper 和 Callback 共享的全局训练进度"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.current_step = 0
            cls._instance.total_steps = 1_500_000
        return cls._instance
    
    @property
    def progress(self):
        return min(self.current_step / max(self.total_steps, 1), 1.0)
    
    def update(self, step):
        self.current_step = step

TRAINING_PROGRESS = TrainingProgress()


# ==========================================
# 1. APF 计算函数
# ==========================================
def calculate_apf_force(host_agent, agents, k_att=1.0, k_rep=0.5, d0=3.0):
    goal_vec = host_agent.goal_global_frame - host_agent.pos_global_frame
    dist_goal = np.linalg.norm(goal_vec)
    
    if dist_goal > 0:
        f_att = k_att * (goal_vec / dist_goal) * host_agent.pref_speed
    else:
        f_att = np.zeros(2)

    f_rep = np.zeros(2)
    for other in agents:
        if other.id == host_agent.id:
            continue
        diff = host_agent.pos_global_frame - other.pos_global_frame
        dist = np.linalg.norm(diff)
        dist_edge = dist - host_agent.radius - other.radius
        if dist_edge <= 0.05:
            dist_edge = 0.05
        if dist_edge < d0:
            mag = k_rep * (1.0 / dist_edge - 1.0 / d0) * (1.0 / (dist_edge ** 2))
            f_rep += mag * (diff / dist)

    return f_att + f_rep


# ==========================================
# 1.5 ORCA 行人动作计算
# ==========================================
def calculate_orca_pedestrian_action(host_agent, agents,
                                     tau=3.0,
                                     max_speed_ratio=0.7,
                                     responsiveness=0.5):
    pos = host_agent.pos_global_frame
    vel = host_agent.vel_global_frame if hasattr(host_agent, 'vel_global_frame') else np.zeros(2)
    goal = host_agent.goal_global_frame
    radius = host_agent.radius
    pref_speed = host_agent.pref_speed * max_speed_ratio

    goal_vec = goal - pos
    dist_goal = np.linalg.norm(goal_vec)
    if dist_goal > 0.1:
        v_pref = (goal_vec / dist_goal) * pref_speed
    else:
        return np.array([0.0, 0.0])

    orca_lines = []

    for other in agents:
        if other.id == host_agent.id:
            continue
        other_pos = other.pos_global_frame
        other_vel = other.vel_global_frame if hasattr(other, 'vel_global_frame') else np.zeros(2)
        other_radius = other.radius
        rel_pos = other_pos - pos
        rel_vel = vel - other_vel
        dist = np.linalg.norm(rel_pos)
        combined_radius = radius + other_radius

        if dist < combined_radius:
            if dist > 0.001:
                push_dir = -rel_pos / dist
            else:
                push_dir = np.array([1.0, 0.0])
            orca_lines.append((np.zeros(2), push_dir))
            continue

        inv_tau = 1.0 / tau
        w = rel_vel - inv_tau * rel_pos
        w_len = np.linalg.norm(w)
        if w_len < 0.0001:
            w_unit = -rel_pos / dist if dist > 0.001 else np.array([1.0, 0.0])
        else:
            w_unit = w / w_len
        line_dir = np.array([-w_unit[1], w_unit[0]])
        u = (combined_radius * inv_tau - w_len) * w_unit
        orca_point = vel + responsiveness * u
        orca_lines.append((orca_point, line_dir))

    v_opt = v_pref.copy()
    for point, direction in orca_lines:
        det = np.cross(direction, v_opt - point)
        if det < 0:
            proj = np.dot(v_opt - point, direction)
            v_opt = point + proj * direction

    speed = np.linalg.norm(v_opt)
    if speed > pref_speed:
        v_opt = v_opt / speed * pref_speed

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
# 全局 scale 计算函数（统一使用，避免多处不一致）
# ==========================================
WARMUP_SCALE = 1.3
SCALE_MAX = 2.0
WARMUP_FRAC = 0.15

def compute_scale(t):
    """
    统一的 scale 计算函数，所有地方都调用这个
    t: 训练进度 [0, 1]
    返回: scale 值
    """
    if t < WARMUP_FRAC:
        return WARMUP_SCALE
    else:
        t_post = (t - WARMUP_FRAC) / (1.0 - WARMUP_FRAC)
        return WARMUP_SCALE + (SCALE_MAX - WARMUP_SCALE) * t_post


# ==========================================
# 🔧 差速驱动参数（与 TurtleBot3 Waffle 一致）
# 只影响 ProgressiveResidualWrapper.action() 中的运动学仿真
# ==========================================
DIFF_DRIVE_MAX_V = 0.22    # 最大线速度 m/s
DIFF_DRIVE_MAX_W = 1.82    # 最大角速度 rad/s
DIFF_DRIVE_DT = 0.1        # 运动学仿真步长（秒）
DIFF_DRIVE_W_GAIN = 3.0    # heading error → 角速度增益


# ==========================================
# 1. 单智能体包装器
# ==========================================
WALL_HALF_SIZE = 6.0  # 12×12 场景

class SimpleSingleAgentWrapper(gym_new.Wrapper):
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

        agents[0].ran_out_of_time = False
        agents[0].time_remaining_to_reach_goal = 999.0
        agents[0].is_at_goal = False
        agents[0].was_at_goal_already = False
        agents[0].is_done = False

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

        all_actions = []
        for i, agent in enumerate(agents):
            if i == 0:
                all_actions.append(action_agent0)
            else:
                action_orca = calculate_orca_pedestrian_action(
                    agent, agents, tau=3.0, max_speed_ratio=0.7, responsiveness=0.5
                )
                all_actions.append(action_orca)
        
        obs, reward_env, done_env, truncated, info = self.env.step(np.array(all_actions))
        self.step_count += 1

        if isinstance(reward_env, (list, tuple, np.ndarray)):
            reward_env = float(reward_env[0])
        
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
        
        if not done and self._check_wall_collision(agent0):
            done = True
            done_reason = "wall_collision"
        
        # ==========================================
        # 奖励函数
        # ==========================================
        if done or truncated:
            if done_reason == "reached_goal" or dist_to_goal < 0.5:
                reward = 10.0
                if self.episode_count % 100 == 0:
                    scale = compute_scale(TRAINING_PROGRESS.progress)
                    print(f"    ✅ Ep {self.episode_count}: 到达目标！"
                          f"(步={self.step_count}, scale={scale:.2f})")
            elif done_reason == "collision":
                reward = -10.0
                if self.episode_count % 100 == 0:
                    print(f"    ❌ Ep {self.episode_count}: 碰撞！(步={self.step_count})")
            elif done_reason == "wall_collision":
                reward = -10.0
                if self.episode_count % 100 == 0:
                    print(f"    🧱 Ep {self.episode_count}: 撞墙！(步={self.step_count})")
            elif truncated:
                reward = -5.0
                if self.episode_count % 100 == 0:
                    print(f"    ⏰ Ep {self.episode_count}: 超时！"
                          f"(步={self.step_count}, 距离={dist_to_goal:.2f})")
            else:
                reward = -5.0
            
            self.episode_count += 1
            self.last_dist = None
        else:
            reward = -0.01
            if self.last_dist is not None:
                dist_change = self.last_dist - dist_to_goal
                reward += dist_change * 0.7
            
            min_dist_edge = self._get_min_dist_to_others(agent0, agents)
            if min_dist_edge < 0.5:
                proximity_penalty = -0.5 * (1.0 - min_dist_edge / 0.5)
                reward += proximity_penalty
            
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
    def __init__(self, env, random_ratio=0.5):
        super().__init__(env)
        self.scenario_type = 0
        self.random_ratio = random_ratio
        self.total_resets = 0
        self.random_count = 0
        self.fixed_count = 0
    
    def _generate_random_scenario(self, raw_env):
        num_agents = len(raw_env.agents)
        spawn_radius = np.random.uniform(2.0, 4.0)
        angles = []
        for _ in range(100):
            angles = sorted(np.random.uniform(0, 2 * np.pi, num_agents))
            min_gap = min(
                min(angles[i+1] - angles[i] for i in range(len(angles)-1)),
                (2 * np.pi - angles[-1] + angles[0])
            )
            if min_gap > np.pi / 4:
                break
        else:
            angles = [i * 2 * np.pi / num_agents + np.random.uniform(-0.3, 0.3)
                      for i in range(num_agents)]
        for i, agent in enumerate(raw_env.agents):
            angle = angles[i]
            x = spawn_radius * np.cos(angle)
            y = spawn_radius * np.sin(angle)
            agent.pos_global_frame = np.array([x, y])
            goal_angle = angle + np.pi + np.random.uniform(-0.5, 0.5)
            goal_radius = np.random.uniform(2.0, 4.0)
            gx = goal_radius * np.cos(goal_angle)
            gy = goal_radius * np.sin(goal_angle)
            agent.goal_global_frame = np.array([gx, gy])
            goal_vec = agent.goal_global_frame - agent.pos_global_frame
            agent.heading_global_frame = np.arctan2(goal_vec[1], goal_vec[0])
            agent.vel_global_frame = np.array([0.0, 0.0])
        self._set_pedestrian_params(raw_env)
    
    def _set_pedestrian_params(self, raw_env):
        raw_env.agents[0].pref_speed = 1.0
        raw_env.agents[0].radius = 0.3
        for i in range(1, len(raw_env.agents)):
            raw_env.agents[i].pref_speed = np.random.uniform(0.8, 1.2)
            raw_env.agents[i].radius = np.random.uniform(0.25, 0.35)
            raw_env.agents[i].vel_global_frame = np.array([0.0, 0.0])
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        raw_env = self.env.unwrapped
        if not hasattr(raw_env, 'agents') or len(raw_env.agents) < 4:
            return obs, info
        self.total_resets += 1
        if np.random.random() < self.random_ratio:
            self._generate_random_scenario(raw_env)
            self.random_count += 1
            if self.total_resets % 200 == 0:
                print(f"    🎲 随机场景 (累计: {self.random_count}/{self.total_resets})")
        else:
            scenario = self.scenario_type % 6
            self._apply_fixed_scenario(raw_env, scenario)
            self._set_pedestrian_params(raw_env)
            self.scenario_type += 1
            self.fixed_count += 1
        try:
            obs = raw_env._get_obs(raw_env.agents)
            if isinstance(obs, dict) and 0 in obs:
                obs = obs[0]
        except:
            pass
        return obs, info
    
    def _apply_fixed_scenario(self, raw_env, scenario):
        if scenario == 0:  # 十字交叉
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
        elif scenario == 1:  # 对角交叉
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
        elif scenario == 2:  # 平行对向
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
        elif scenario == 3:  # 扇形汇聚
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
        elif scenario == 4:  # 星形散开
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
        else:  # 斜角混战
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
# 2. 渐进式残差放大（两阶段）
# 🔧 唯一改动：action() 末尾加入差速运动学仿真
# ==========================================
class ProgressiveResidualWrapper(gym_new.ActionWrapper):
    """
    两阶段残差：
    - 前 warmup_frac: scale 固定为 warmup_scale（=1.3，和 v7 一样能学）
    - 之后: 从 warmup_scale 线性增长到 scale_max
    
    🔧 v8b_dd 改动：
    SAC 算出 (final_speed, final_heading) 后，不再直接给环境执行，
    而是经过差速运动学仿真：
      1. heading_error = final_heading - 当前朝向
      2. w = clip(gain * heading_error, -max_w, max_w)
      3. 转弯时线速度衰减（大角度 ×0.3，中角度 ×0.6）
      4. 新朝向 = 当前朝向 + w * dt
      5. 从仿真结果反算 (speed, heading) 给环境
    
    这样 RL 在训练时就能感受到「转弯需要时间、转弯时会减速」，
    部署到差速机器人时策略天然适配。
    """
    
    def __init__(self, env,
                 warmup_scale=WARMUP_SCALE,
                 scale_max=SCALE_MAX,
                 warmup_frac=WARMUP_FRAC,
                 ):
        super().__init__(env)
        
        self.action_space = gym_new.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        self.warmup_scale = warmup_scale
        self.scale_max = scale_max
        self.warmup_frac = warmup_frac
        
        self._last_scale = warmup_scale
        self._log_interval = 50000
        self._last_log_step = 0
    
    def _get_scale(self):
        """调用统一的 scale 计算函数"""
        scale = compute_scale(TRAINING_PROGRESS.progress)
        self._last_scale = scale
        return scale
    
    def action(self, action_sac):
        raw_env = self.env.unwrapped
        try:
            agent0 = raw_env.agents[0]
            agents = raw_env.agents
        except:
            return np.array([0.5, 0.0])
        
        vec_apf = calculate_apf_force(agent0, agents)
        speed_apf = np.linalg.norm(vec_apf)
        heading_apf = np.arctan2(vec_apf[1], vec_apf[0])
        
        scale = self._get_scale()

        delta_speed = action_sac[0] * scale * agent0.pref_speed
        delta_heading = action_sac[1] * scale
        
        final_speed = np.clip(speed_apf + delta_speed, 0, agent0.pref_speed)
        final_heading = (heading_apf + delta_heading + np.pi) % (2 * np.pi) - np.pi
        
        # =============================================
        # 🔧 差速运动学仿真（v8b 唯一改动）
        #
        # 不再直接 return [final_speed, final_heading]
        # 而是模拟差速机器人从当前朝向转向 final_heading 的过程
        # =============================================
        current_heading = agent0.heading_global_frame
        heading_error = (final_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
        
        # 角速度（与部署代码一致）
        w = np.clip(DIFF_DRIVE_W_GAIN * heading_error, -DIFF_DRIVE_MAX_W, DIFF_DRIVE_MAX_W)
        
        # 线速度：映射到差速机器人范围，转弯时衰减
        v = np.clip(final_speed * (DIFF_DRIVE_MAX_V / agent0.pref_speed), 0, DIFF_DRIVE_MAX_V)
        abs_herr = abs(heading_error)
        # if abs_herr > 1.2:
        #     v *= 0.3    # 大角度：慢速前进 + 转向
        # elif abs_herr > 0.5:
        #     v *= 0.6    # 中角度：适度减速
        
        # 仿真一步：算出新朝向
        new_heading = (current_heading + w * DIFF_DRIVE_DT + np.pi) % (2 * np.pi) - np.pi
        
        # 反算回环境需要的 (speed, heading) 格式
        actual_speed = v / (DIFF_DRIVE_MAX_V / agent0.pref_speed) if DIFF_DRIVE_MAX_V > 0 else 0
        actual_heading = new_heading
        
        # 日志（保持原有格式）
        current_step = TRAINING_PROGRESS.current_step
        if current_step - self._last_log_step >= self._log_interval:
            progress_pct = TRAINING_PROGRESS.progress * 100
            phase = "暖身" if TRAINING_PROGRESS.progress < self.warmup_frac else "成长"
            print(f"    🔧 [{phase}] 进度={progress_pct:.1f}% | scale={scale:.3f} | "
                  f"APF=[{speed_apf:.2f}, {np.degrees(heading_apf):.0f}°] | "
                  f"δ=[{delta_speed:+.2f}, {np.degrees(delta_heading):+.0f}°] | "
                  f"herr={np.degrees(heading_error):.0f}° v_dd={v:.3f}")
            self._last_log_step = current_step
        
        return np.array([actual_speed, actual_heading])


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
# 3.5 实时进度回调（同步全局进度）
# ==========================================
class RealTimeProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, check_freq=1000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.last_time = None
        self.start_time = None
        TRAINING_PROGRESS.total_steps = total_timesteps
    
    def _on_training_start(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        print("\n🚀 训练开始！")
        print(f"   两阶段残差：前15% scale=1.3 固定，之后线性增长到 2.0")
        print(f"   🔧 差速驱动仿真：max_v={DIFF_DRIVE_MAX_V}, max_w={DIFF_DRIVE_MAX_W}, dt={DIFF_DRIVE_DT}")
    
    def _on_step(self) -> bool:
        # 每步更新全局进度
        TRAINING_PROGRESS.update(self.num_timesteps)
        
        if self.num_timesteps % self.check_freq == 0:
            now = time.time()
            elapsed = now - self.last_time
            fps = self.check_freq / elapsed if elapsed > 0 else 0
            progress = self.num_timesteps / self.total_timesteps * 100
            total_elapsed = now - self.start_time
            
            # 使用统一的 scale 计算函数
            t = self.num_timesteps / self.total_timesteps
            scale = compute_scale(t)
            
            if self.num_timesteps > 0:
                eta_seconds = (self.total_timesteps - self.num_timesteps) * total_elapsed / self.num_timesteps
                eta_minutes = int(eta_seconds / 60)
                print(f"⏳ {progress:.1f}% | {self.num_timesteps}/{self.total_timesteps} | "
                      f"FPS:{fps:.0f} | scale={scale:.2f} | 剩余:{eta_minutes}min")
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
    
    env = UniversalGymAdapter(env)
    env = TimeLimit(env, max_episode_steps=400)
    env = SimpleSingleAgentWrapper(env)
    env = MixedScenarioWrapper(env, random_ratio=0.3)
    
    env = ProgressiveResidualWrapper(
        env,
        warmup_scale=WARMUP_SCALE,
        scale_max=SCALE_MAX,
        warmup_frac=WARMUP_FRAC,
    )
    
    env = FlattenObservationWrapper(env)
    env = Monitor(env)
    return env


if __name__ == "__main__":
    log_dir = "./sac_apf_logs_v8b_dd/"
    os.makedirs(log_dir, exist_ok=True)
    
    total_steps = 1_500_000
    TRAINING_PROGRESS.total_steps = total_steps
    
    print("=" * 60)
    print("🎯 SAC + 渐进式残差 APF (v8b_dd - Diff-Drive)")
    print("=" * 60)
    print()
    print("  v8b_dd = v8b + 差速运动学仿真层")
    print("  ─────────────────────────────────────────────")
    print("  唯一改动：ProgressiveResidualWrapper.action()")
    print("  SAC 算出 (speed, heading) 后经过差速仿真再给环境")
    print("  RL 在训练时就感受到：转弯需要时间、转弯时会减速")
    print("  ─────────────────────────────────────────────")
    print(f"  差速参数: max_v={DIFF_DRIVE_MAX_V}m/s, max_w={DIFF_DRIVE_MAX_W}rad/s")
    print(f"  仿真步长: dt={DIFF_DRIVE_DT}s, 增益={DIFF_DRIVE_W_GAIN}")
    print("  ─────────────────────────────────────────────")
    print("  渐进式残差（与 v8b 完全相同）：")
    print("  阶段1: t < 15%  → scale = 1.3（固定暖身）")
    print("  阶段2: t ≥ 15%  → scale 从 1.3 线性增长到 2.0")
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
    
    print(f"\n>>> 开始训练... (总步数: {total_steps})")
    
    progress_callback = RealTimeProgressCallback(
        total_timesteps=total_steps,
        check_freq=5000
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=log_dir,
        name_prefix='sac_v8b_dd'
    )
    
    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_callback, progress_callback],
        log_interval=10
    )
    
    model.save("sac_v8b_dd_final")
    print(f"\n>>> ✅ 训练完成！最终 scale = {compute_scale(1.0):.3f}")