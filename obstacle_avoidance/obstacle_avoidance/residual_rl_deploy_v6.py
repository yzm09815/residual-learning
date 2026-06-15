#!/usr/bin/env python3
"""
v20: 距离自适应切换 — 远处纯 APF，近障碍物引入 RL 残差
纯局部规划
核心策略：
  ✅ 空旷区域（front_min > 1.5m）：纯 APF 控制（已验证完美）
  ✅ 接近障碍物（front_min < 0.8m）：APF + RL 残差混合
  ✅ 中间距离：alpha 线性过渡
  ✅ RL 残差被限制在小范围内（heading ±34°, speed ±0.3），防止绕圈
  ✅ 使用 v8b 1.5M 步模型（训练时 ep_rew_mean=+10）

使用 v8b 模型而非重新训练的原因：
  v8b 在训练环境中已经学会了避障策略（成功率高）
  全程开启时因为全向→差速的 gap 导致绕圈
  但局部小幅修正时（alpha<0.5, dh<±34°），修正量足够小
  差速机器人完全可以执行，不会产生问题

其余保留：
  ✅ Ground Truth 位置（替代 odom）
  ✅ obs 按字母序排列（与训练一致）
  ✅ 108维 = agent0(27) × 4 agents
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
import numpy as np
import math
import subprocess  # ✅ v19 新增：用于启动 gz topic 子进程
import re           # ✅ v19 新增：用于解析 gz topic 输出
import threading    # ✅ v19 新增：后台线程读取 ground truth

from stable_baselines3 import SAC

#角度归一化，把任何角度都强制转换并限制在 $[-\pi, \pi]$
#角度如果离谱（比如 $1000\pi$），循环要跑几百次，极慢。
#只能处理单个数字（标量）。如果传进一个数组，程序会报错。
def wrap(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


#角度归一化，把任何角度都强制转换并限制在 $[-\pi, \pi]$
#无论角度多离谱，只要做一次除法求余，瞬间出结果，完美支持批量运算，angle可以为数组
def wrap_train(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class ResidualRLDeployNode(Node):
    def __init__(self):
        super().__init__('residual_rl_deploy_node')

        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(Float32MultiArray, '/pedestrian_states', self.ped_cb, 10)

        self.target_x = 4.0
        self.target_y = 4.0
        self.pref_speed = 1.0
        self.radius = 0.3

        # APF 参数 — 与训练 calculate_apf_force() 一致
        self.k_att = 1.0        #Attraction
        self.k_rep = 0.5        #Repulsion 斥力
        self.d0_ped = 3.0      # ✅ 训练代码 d0=3.0
        self.d0_wall = 1.0
        self.k_rep_wall = 0.5

        self.max_other = 3 #最大感知障碍物数量 每个 agent 的 obs = 6 维自身状态 + 3×7 = 21 维他人状态 = 27 维
        self.max_v = 0.22 #最大线速度
        self.max_w = 1.82 #最大角速度

        #c：Current
        self.cx = 0.0
        self.cy = 0.0
        self.cyaw = 0.0 #偏航角（Yaw），也就是车头指着哪个方向。
        # $X$ 轴和 $Y$ 轴方向的线速度分量
        self.cvx = 0.0
        self.cvy = 0.0
        self.prev_w = 0.0 #上一时刻的角速度（Previous $\omega$）。

        # ===== ✅ v19 新增：Ground Truth 相关 =====
        # 后台线程通过 gz topic -e 持续监听 Gazebo 的 dynamic_pose/info
        # 从中解析 waffle（机器人）的真实世界坐标，替代 odom 的位置
        # 这样 cx, cy, cyaw 就是真实值，不会有轮式里程计的累积漂移
        self.gt_ready = False  # 是否已收到过 ground truth 数据
        self._gt_lock = threading.Lock()  # 线程锁，保护 cx/cy/cyaw 的并发读写
        self._start_gz_listener()  # 启动后台监听线程

        # 逃逸
        self.stuck_count = 0  #卡死计时器
        self.last_dist = 999.0  #记录上一时刻距离目标的距离
        self.stuck_threshold = 100 #卡死计时极限
        self.escape_strength = 2.0  #逃逸力度
        self.escape_dir = 1.0  #direction 逃逸方向

        # 行人状态 [x, y, vx, vy, speed, goal_x, goal_y, pref_speed, radius]
        self.ped_states = [
            [1.0, 0.0, 0.0, 0.0, 0.8, 1.0, 4.0, 0.8, 0.3],
            [-1.0, 2.0, 0.0, 0.0, 0.6, 3.0, 2.0, 0.6, 0.3],
            [2.0, -2.0, 0.0, 0.0, 0.7, -3.0, -2.0, 0.7, 0.3],
        ]
        self.ped_received = False
        #每跑 20 次（_diag_interval = 20）才在屏幕上打印一次日志
        self._diag_cnt = 0
        self._diag_interval = 20

        # RL 模型
        self.use_rl = False
        # model_path = '/mnt/c/Users/admin/Desktop/训练模型/wall/2_动态残差_30%0.7_sac_1.3_1.3_12x12wall/sac_progressive_v8b_1500000_steps'
        # try:
        #     self.model = SAC.load(model_path)
        #     dim = self.model.observation_space.shape[0]
        #     self.get_logger().info(f'🧠 SAC 加载成功 | obs={dim}维')
        #     self.use_rl = True
        # except Exception as e:
        #     self.get_logger().error(f'❌ 模型加载失败: {e}')
        #     self.use_rl = False

        self.get_logger().info(f'🎯 目标:({self.target_x},{self.target_y}) | RL={"ON" if self.use_rl else "OFF"}')
        self.get_logger().info(f'📐 APF: k_att={self.k_att}, k_rep={self.k_rep}, d0={self.d0_ped}')
        self.get_logger().info(f'📋 OBS 顺序: 字母序 [dist_to_goal, heading_ego, is_learning, num_other, others(21), pref_speed, radius] × 4 agents')
        self.get_logger().info(f'🌍 v20: 距离自适应切换（远=纯APF，近=APF+RL）')

    # =========================================================
    #  ✅ v19 新增：Ground Truth 后台线程
    #  通过 gz topic -e 持续读取 Gazebo 的 dynamic_pose/info
    #  从中解析 waffle 的真实世界坐标 (x, y, yaw)
    #  这样机器人坐标和行人坐标（来自 pedestrian_mover 的世界坐标）
    #  就在同一个坐标系下，彻底消除 odom 漂移导致的 7 米偏差问题
    # =========================================================
    def _start_gz_listener(self):
        """启动后台守护线程，持续监听 gz topic"""
        self._gz_thread = threading.Thread(target=self._gz_listener_loop, daemon=True)
        self._gz_thread.start()
        self.get_logger().info('🛰️ Ground Truth 监听线程已启动')

    def _gz_listener_loop(self):
        """
        后台循环：运行 gz topic -e 持续读取 dynamic_pose/info
        这个 topic 包含 Gazebo 世界中所有模型的真实位姿，
        以 Pose_V 消息格式发布，每帧包含多个 pose { ... } 块。
        我们只需要找到 name: "waffle" 的那个块，提取位置和朝向。
        """
        cmd = [
            'gz', 'topic', '-e', '-t',
            '/world/training_world_12x12/dynamic_pose/info'
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1
            )
            self.get_logger().info('🛰️ gz topic -e 进程已启动')

            # 逐行读取，按空行或 "---" 分隔成完整的消息帧
            buffer_lines = []
            for line in proc.stdout:
                stripped = line.strip()

                # 空行或分隔符表示一帧结束
                if stripped == '' or stripped == '---':
                    if buffer_lines:
                        frame_text = '\n'.join(buffer_lines)
                        self._parse_frame(frame_text)
                        buffer_lines = []
                    continue

                buffer_lines.append(stripped)

        except Exception as e:
            self.get_logger().error(f'🛰️ gz listener 异常: {e}')

    def _parse_frame(self, frame_text):
        """
        解析一帧 Pose_V 消息，找到 waffle 的位置。
        消息格式类似：
          pose {
            name: "waffle"
            position { x: 1.23  y: 4.56  z: 0.01 }
            orientation { x: 0  y: 0  z: 0.5  w: 0.866 }
          }
          pose {
            name: "pedestrian_1"
            ...
          }
        """
        lines = frame_text.split('\n')

        # 找到包含 "waffle" 的行
        waffle_idx = -1
        for i, line in enumerate(lines):
            if '"waffle"' in line:
                waffle_idx = i
                break

        if waffle_idx < 0:
            return  # 这帧没有 waffle 信息

        # 从 waffle 行附近搜索 position 和 orientation（前5行到后15行）
        search_start = max(0, waffle_idx - 5)
        search_end = min(len(lines), waffle_idx + 15)
        block = '\n'.join(lines[search_start:search_end])

        # 提取 position 中的 x, y
        pos_x = self._extract_field(block, 'position', 'x')
        pos_y = self._extract_field(block, 'position', 'y')

        # 提取 orientation 四元数
        ori_x = self._extract_field(block, 'orientation', 'x')
        ori_y = self._extract_field(block, 'orientation', 'y')
        ori_z = self._extract_field(block, 'orientation', 'z')
        ori_w = self._extract_field(block, 'orientation', 'w')

        if pos_x is not None and pos_y is not None:
            # 计算偏航角 yaw（从四元数转换），与 odom_cb 中的公式一致
            gyaw = self.cyaw  # fallback：用当前值
            if all(v is not None for v in [ori_x, ori_y, ori_z, ori_w]):
                gyaw = math.atan2(2.0 * (ori_w * ori_z + ori_x * ori_y),
                                  1.0 - 2.0 * (ori_y * ori_y + ori_z * ori_z))

            # 用线程锁保护写入，避免与主线程（scan_cb）读取冲突
            with self._gt_lock:
                self.cx = pos_x
                self.cy = pos_y
                self.cyaw = gyaw
                if not self.gt_ready:
                    self.gt_ready = True
                    self.get_logger().info(
                        f'🛰️ 首次收到 Ground Truth: ({pos_x:.3f}, {pos_y:.3f}), yaw={math.degrees(gyaw):.1f}°'
                    )

    def _extract_field(self, block, section, field):
        """
        从文本块中提取嵌套字段值。
        例如从 "position { x: 1.23 y: 4.56 z: 0.01 }" 中提取 x 的值 1.23。
        支持多行和单行两种格式。
        """
        # 找到 section（如 position）的起始 {
        sec_match = re.search(rf'{section}\s*\{{', block)
        if not sec_match:
            return None

        # 从 { 开始，找到匹配的 }（处理嵌套大括号）
        start = sec_match.end()
        brace_count = 1
        end = start
        while end < len(block) and brace_count > 0:
            if block[end] == '{':
                brace_count += 1
            elif block[end] == '}':
                brace_count -= 1
            end += 1

        section_content = block[start:end]

        # 在 section 内容中找 field: value（支持科学计数法）
        field_match = re.search(rf'{field}:\s*([-\d.eE+]+)', section_content)
        if field_match:
            try:
                return float(field_match.group(1))
            except ValueError:
                return None
        return None

    def cmd(self, v, w):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_footprint' 
        m.twist.linear.x = float(v)
        m.twist.angular.z = float(w)
        self.pub.publish(m)

    # ✅ v19 修改：odom_cb 只取速度，不取位置和 yaw
    # 位置和 yaw 由 ground truth 线程更新（真实世界坐标，无漂移）
    # 速度仍从 odom 取，因为瞬时速度是准确的，不受累积漂移影响
    def odom_cb(self, msg): #cb  callback 回调函数
        # ❌ v19: 不再从 odom 取位置（有累积漂移，长距离后会差好几米）：
        # self.cx = msg.pose.pose.position.x
        # self.cy = msg.pose.pose.position.y
        # q = msg.pose.pose.orientation #q就是四元数
        # self.cyaw = math.atan2(2.0*(q.w*q.z+q.x*q.y), 1.0-2.0*(q.y*q.y+q.z*q.z))

        # ✅ v19: 只取速度（瞬时值，不受漂移影响）
        # $X$ 轴和 $Y$ 轴方向的线速度分量
        self.cvx = msg.twist.twist.linear.x
        self.cvy = msg.twist.twist.linear.y

    def ped_cb(self, msg):
        if len(msg.data) >= 27:
            for i in range(3):
                base = i * 9
                self.ped_states[i] = [float(msg.data[base + j]) for j in range(9)]
            if not self.ped_received:
                self.ped_received = True
                self.get_logger().info('🚶 收到行人真实数据！')

    # =========================================================
    #  APF — 与训练代码完全一致
    # =========================================================
    def calculate_apf_force_aligned(self):
        pos = np.array([self.cx, self.cy])
        goal = np.array([self.target_x, self.target_y])
        goal_vec = goal - pos
        dist_goal = np.linalg.norm(goal_vec)
        if dist_goal > 0:
            f_att = self.k_att * (goal_vec / dist_goal) * self.pref_speed #引力系数x期望速度
        else:
            f_att = np.zeros(2) #生成0，0

        f_rep = np.zeros(2)
        for i in range(3):
            ps = self.ped_states[i]
            other_pos = np.array([ps[0], ps[1]])
            other_radius = ps[8]
            diff = pos - other_pos
            dist = np.linalg.norm(diff)
            dist_edge = dist - self.radius - other_radius
            if dist_edge <= 0.05:
                dist_edge = 0.05
            if dist_edge < self.d0_ped:
                mag = self.k_rep * (1.0 / dist_edge - 1.0 / self.d0_ped) * (1.0 / (dist_edge ** 2))
                f_rep += mag * (diff / dist)

        return f_att + f_rep

    def calculate_wall_repulsion(self, msg):
        pos = np.array([self.cx, self.cy])
        f_rep_wall = np.zeros(2)
        front_min = 5.0
        ranges = np.array(msg.ranges)
        for i, r in enumerate(ranges):
            if math.isnan(r) or math.isinf(r) or r <= 0:
                continue
            sr = max(float(r), 0.15)
            al = msg.angle_min + i * msg.angle_increment
            if al > math.pi:
                al -= 2 * math.pi
            if abs(al) < 0.4:
                front_min = min(front_min, sr)
            if sr < self.d0_wall:
                ag = al + self.cyaw
                op = pos + sr * np.array([math.cos(ag), math.sin(ag)])
                is_ped = False
                for j in range(3):
                    pp = np.array([self.ped_states[j][0], self.ped_states[j][1]])
                    if np.linalg.norm(op - pp) < 0.5:
                        is_ped = True
                        break
                if is_ped:
                    continue
                diff = pos - op
                d = np.linalg.norm(diff)
                de = max(d - self.radius - 0.3, 0.05)
                if de < self.d0_wall and d > 0.001:
                    mag = self.k_rep_wall * (1.0/de - 1.0/self.d0_wall) / (de**2)
                    mag = min(mag, 5.0)
                    f_rep_wall += mag * (diff / d)
        #f_rep_wall (墙壁总斥力向量)
        #front_min (正前方最小安全距离)
        return f_rep_wall, front_min 

    # =========================================================
    #  OBS 构建辅助函数
    # =========================================================
    #将机器人（Host）和行人（Other Agent）在全局坐标系（笛卡尔坐标系）下的
    # 原始位置、速度，转换成以机器人为中心的局部参考系（通常是相对于目标点方向）下的相对坐标
    def _get_ref(self, pos, goal):
        goal_direction = goal - pos
        dist_to_goal = math.sqrt(goal_direction[0]**2 + goal_direction[1]**2)
        if dist_to_goal > 1e-8:
            ref_prll = goal_direction / dist_to_goal
        else:
            ref_prll = goal_direction.copy()
        ref_orth = np.array([-ref_prll[1], ref_prll[0]])
        return ref_prll, ref_orth, dist_to_goal

    def _get_heading_ego(self, heading_global, ref_prll):
        #np.arctan2(y, x) 是一个数学函数，它能算出这个向量在全局地图上的角度
        #ref_prll[0]: 存储的是方向向量在 $x$ 轴（通常是地图的横向）上的投影长度。
        #ref_prll[1]: 存储的是方向向量在 $y$ 轴（通常是地图的横向）上的投影长度。
        ref_prll_angle = np.arctan2(ref_prll[1], ref_prll[0])
        return wrap(heading_global - ref_prll_angle)

    def _compute_other_agent_obs(self, host_pos, host_radius, host_ref_prll, host_ref_orth,
                                  other_pos, other_vel, other_radius):
        rel_pos = other_pos - host_pos
        #dot 模长x cos 位置投影/速度投影
        p_prll = np.dot(rel_pos, host_ref_prll) #纵向距离（Parallel）。
        p_orth = np.dot(rel_pos, host_ref_orth) #横向偏移（Orthogonal）。
        v_prll = np.dot(other_vel, host_ref_prll)
        v_orth = np.dot(other_vel, host_ref_orth)
        dist_center = np.linalg.norm(rel_pos)
        combined_radius = host_radius + other_radius
        dist_edge = dist_center - combined_radius
        return np.array([p_prll, p_orth, v_prll, v_orth,
                         other_radius, combined_radius, dist_edge], dtype=np.float32)

    # =========================================================
    #  构建单个 agent 的 27 维 obs — ✅ 字母序排列
    # =========================================================
    def build_single_agent_obs_alphabetical(self, agent_pos, agent_goal, agent_heading,
                                             agent_vel, agent_pref_speed, agent_radius,
                                             is_learning_val, other_agents_info):
        """
        构建单个 agent 的 27 维观测，按字母序排列：
          [0]    dist_to_goal
          [1]    heading_ego_frame
          [2]    is_learning
          [3]    num_other_agents
          [4:25] other_agents_states (3×7=21)
          [25]   pref_speed
          [26]   radius

        other_agents_info: list of (pos, vel, radius) for other agents
        """
        pos = np.array(agent_pos)
        goal = np.array(agent_goal)
        ref_prll, ref_orth, dist_to_goal = self._get_ref(pos, goal)
        heading_ego = self._get_heading_ego(agent_heading, ref_prll)

        # other_agents_states: 按距离排序（closest_first）
        other_states = np.zeros((self.max_other, 7), dtype=np.float32)
        num_observed = 0

        # 计算距离并排序 找出离的最近的
        dists = []
        for idx, (o_pos, o_vel, o_rad) in enumerate(other_agents_info):
            d = np.linalg.norm(np.array(o_pos) - pos)
            dists.append((idx, d))
        dists.sort(key=lambda x: x[1])

        for rank, (idx, _) in enumerate(dists):
            if rank >= self.max_other:
                break
            o_pos, o_vel, o_rad = other_agents_info[idx]
            obs_7 = self._compute_other_agent_obs(
                pos, agent_radius, ref_prll, ref_orth,
                np.array(o_pos), np.array(o_vel), o_rad
            )
            other_states[rank] = obs_7
            num_observed += 1

        # ✅ 按字母序排列！
        obs = np.concatenate([
            np.array([dist_to_goal], dtype=np.float32),          # dist_to_goal
            np.array([heading_ego], dtype=np.float32),           # heading_ego_frame
            np.array([is_learning_val], dtype=np.float32),       # is_learning
            np.array([float(num_observed)], dtype=np.float32),   # num_other_agents
            other_states.flatten(),                               # other_agents_states (21)
            np.array([agent_pref_speed], dtype=np.float32),      # pref_speed
            np.array([agent_radius], dtype=np.float32),          # radius
        ])
        return obs

    # =========================================================
    #  构建完整 108 维 state
    # =========================================================
    def build_full_state(self):
        """
        构建 108 维 state = agent0(27) + agent1(27) + agent2(27) + agent3(27)

        agent0 = 机器人（is_learning 看 policy.str == "learning"）
        agent1/2/3 = 行人

        注意：训练时 is_learning 的值来自 get_agent_data_equiv("policy.str", "learning")
        所有 agent 用的都是 ExternalPolicy，其 policy.str = "External"
        所以 is_learning = (External == learning) = False = 0.0

        但训练代码中 SimpleSingleAgentWrapper.reset() 设了：
            agents[0].is_learning = True
        然而 is_learning 在 obs 中的 attr 是：
            'get_agent_data_equiv("policy.str", "learning")'
        这读的是 policy.str，不是 agent.is_learning 属性！
        所以 is_learning 在 obs 中实际上是 0.0（因为 External != learning）

        除非环境在某处修改了 policy.str... 需要确认。
        先按 False (0.0) 处理。
        """
        robot_pos = [self.cx, self.cy]
        robot_goal = [self.target_x, self.target_y]
        robot_heading = self.cyaw
        robot_speed = math.hypot(self.cvx, self.cvy)
        robot_vel = [robot_speed * math.cos(self.cyaw),
                     robot_speed * math.sin(self.cyaw)]

        # 收集所有 agent 信息
        all_agents = []  # (pos, goal, heading, vel, pref_speed, radius, is_learning)

        # Agent 0: 机器人
        all_agents.append({
            'pos': robot_pos,
            'goal': robot_goal,
            'heading': robot_heading,
            'vel': robot_vel,
            'pref_speed': self.pref_speed,
            'radius': self.radius,
            'is_learning': 0.0,  # ExternalPolicy.str != "learning"
        })

        # Agent 1/2/3: 行人
        for i in range(3):
            ps = self.ped_states[i]
            ped_vel = [ps[2], ps[3]]
            if abs(ps[2]) + abs(ps[3]) > 0.01:
                ped_heading = math.atan2(ps[3], ps[2])
            else:
                goal_dir = np.array([ps[5], ps[6]]) - np.array([ps[0], ps[1]])
                ped_heading = math.atan2(goal_dir[1], goal_dir[0])
            all_agents.append({
                'pos': [ps[0], ps[1]],
                'goal': [ps[5], ps[6]],
                'heading': ped_heading,
                'vel': ped_vel,
                'pref_speed': ps[7],
                'radius': ps[8],
                'is_learning': 0.0,
            })

        # 为每个 agent 构建 27 维 obs
        all_obs = []
        for agent_idx in range(4):
            agent = all_agents[agent_idx]
            # other_agents = 除自己以外的所有 agent
            others = []
            for j in range(4):
                if j == agent_idx:
                    continue
                other = all_agents[j]
                others.append((other['pos'], other['vel'], other['radius']))

            obs_27 = self.build_single_agent_obs_alphabetical(
                agent_pos=agent['pos'],
                agent_goal=agent['goal'],
                agent_heading=agent['heading'],
                agent_vel=agent['vel'],
                agent_pref_speed=agent['pref_speed'],
                agent_radius=agent['radius'],
                is_learning_val=agent['is_learning'],
                other_agents_info=others,
            )
            all_obs.append(obs_27)

        return np.concatenate(all_obs)  # 108 维

    def _run_diagnostics(self, state, apf_speed, apf_heading):
        self._diag_cnt += 1
        if self._diag_cnt % self._diag_interval != 0:
            return

        self.get_logger().info('')
        self.get_logger().info('=' * 50)
        self.get_logger().info('====== OBS 诊断 (字母序 + Ground Truth) ======')
        self.get_logger().info(f'  state 范围: [{state.min():.3f}, {state.max():.3f}]')

        # Agent 0 的字母序 obs
        a0 = state[:27]
        self.get_logger().info(f'  agent0:')
        self.get_logger().info(f'    [0]  dist_to_goal:      {a0[0]:.3f}')
        self.get_logger().info(f'    [1]  heading_ego_frame: {a0[1]:.3f} ({math.degrees(a0[1]):.1f}°)')
        self.get_logger().info(f'    [2]  is_learning:       {a0[2]:.1f}')
        self.get_logger().info(f'    [3]  num_other_agents:  {a0[3]:.1f}')
        for i in range(3):
            b = 4 + i * 7
            self.get_logger().info(f'    other[{i}]: p=({a0[b]:.2f},{a0[b+1]:.2f}) v=({a0[b+2]:.2f},{a0[b+3]:.2f}) de={a0[b+6]:.2f}')
        self.get_logger().info(f'    [25] pref_speed:        {a0[25]:.3f}')
        self.get_logger().info(f'    [26] radius:            {a0[26]:.3f}')
        self.get_logger().info(f'  APF(对齐): speed={apf_speed:.3f} heading={math.degrees(apf_heading):.1f}°')
        self.get_logger().info(f'  robot GT: ({self.cx:.2f},{self.cy:.2f}) yaw={math.degrees(self.cyaw):.1f}°')
        self.get_logger().info('=' * 50)

    # =========================================================
    #  主控制循环
    # =========================================================
    def scan_cb(self, msg):
        # ✅ v19: 等待 ground truth 数据就绪才开始控制
        if not self.gt_ready:
            self.cmd(0.0, 0.0)
            return
        # 没收到行人数据前不动
        if not self.ped_received:
            self.cmd(0.0, 0.0)
            return
        # 碰撞检测：与任何行人距离 < 两者半径之和
        for i in range(3):
            ps = self.ped_states[i]
            d = math.hypot(ps[0] - self.cx, ps[1] - self.cy)
            if d < (self.radius + ps[8]):
                self.get_logger().info(f'💥 碰撞行人 {i}！停止')
                self.cmd(0.0, 0.0)
                return
        dx = self.target_x - self.cx
        dy = self.target_y - self.cy
        dist = math.hypot(dx, dy)

        if dist < 0.3:
            # ✅ v19: 到达时打印 GT 坐标，现在和 Gazebo 显示一致
            self.get_logger().info(f'🎉 到达！GT位置:({self.cx:.2f},{self.cy:.2f})')
            self.cmd(0.0, 0.0)
            self.stuck_count = 0
            return

        # APF（与训练一致）
        f_apf = self.calculate_apf_force_aligned()
        f_wall, front_min = self.calculate_wall_repulsion(msg)

        spd_apf = np.linalg.norm(f_apf)
        hd_apf = math.atan2(f_apf[1], f_apf[0])

        # 逃逸
        if abs(dist - self.last_dist) < 0.03:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        self.last_dist = dist

        escape_active = False
        f_escape = np.zeros(2)
        # if self.stuck_count > self.stuck_threshold:
        #     escape_active = True
        #     gv = np.array([dx, dy])
        #     gv_unit = gv / dist
        #     tangent = np.array([-gv_unit[1], gv_unit[0]]) * self.escape_dir
        #     f_escape = self.escape_strength * tangent
        #     if self.stuck_count > self.stuck_threshold * 3:
        #         self.escape_dir *= -1.0 
        #         self.stuck_count = self.stuck_threshold + 1

        # RL 残差
        # =========================================================
        # ✅ v20: 距离自适应切换 — 远处纯 APF，近障碍物引入 RL
        #
        # 核心逻辑：用 front_min（激光雷达前方最近障碍物距离）判断
        #   - front_min > 1.5m: 纯 APF（空旷区域 APF 已经完美）
        #   - front_min < 0.8m: RL 权重最大（需要精确避障）
        #   - 中间线性过渡
        #
        # 同时考虑最近行人距离（行人不一定在正前方）
        # 取两者的较小值作为 "威胁距离"
        #
        # 为什么用 front_min 而不是只用行人距离：
        #   激光雷达能同时检测墙壁、圆柱、行人等所有障碍物
        #   这样 RL 在墙壁附近也会被激活，帮助解决 APF 局部极小值
        # =========================================================
        rl_info = 'APF'
        if self.use_rl and self.ped_received:
            state = self.build_full_state()
            self._run_diagnostics(state, spd_apf, hd_apf)
            action, _ = self.model.predict(state, deterministic=True)

            # v8b 训练结束时 scale=2.0，用完整 scale 让 RL 发挥最大能力
            ds_raw = action[0] * 2.0 * self.pref_speed
            dh_raw = action[1] * 2.0

            # --- 计算威胁距离: 取激光雷达前方 和 最近行人 的较小值 ---
            min_ped_dist = float('inf')
            for i in range(3):
                ps = self.ped_states[i]
                d = math.hypot(ps[0] - self.cx, ps[1] - self.cy) - self.radius - ps[8]
                min_ped_dist = min(min_ped_dist, d)

            threat_dist = min(front_min, min_ped_dist)

            # alpha: RL 权重，威胁越近越大（启动条件不变）
            alpha_max = 0.8   # RL 最大权重（加大，让 RL 在障碍物附近有足够话语权）
            dist_far = 1.5    # 超过此距离 RL 不参与（纯 APF）— 不变
            dist_near = 0.6   # 低于此距离 RL 权重最大 — 不变
            if threat_dist >= dist_far:
                alpha = 0.0
            elif threat_dist <= dist_near:
                alpha = alpha_max
            else:
                alpha = alpha_max * (dist_far - threat_dist) / (dist_far - dist_near)

            # 限制 RL 修正幅度（加大范围，让 RL 能大幅转向绕过障碍物）
            dh_clipped = float(np.clip(dh_raw, -1.2, 1.2))   # 最大 ±69°（之前 ±34° 太小绕不过去）
            ds_clipped = float(np.clip(ds_raw, -0.5 * self.pref_speed, 1.5 * self.pref_speed))

            # 加权混合：APF + alpha * RL残差
            f_spd = float(np.clip(spd_apf + alpha * ds_clipped, 0.10 * self.pref_speed, self.pref_speed))
            f_hd = wrap_train(hd_apf + alpha * dh_clipped)

            if alpha > 0.01:
                rl_info = f'AI:[{action[0]:.2f},{action[1]:.2f}] α={alpha:.2f} th={threat_dist:.1f}'
            else:
                rl_info = 'APF'
        else:
            alpha = 0.0  # 纯 APF 模式，RL 不参与
            f_total = f_apf + f_wall + f_escape
            spd_total = np.linalg.norm(f_total)
            hd_total = math.atan2(f_total[1], f_total[0])
            f_spd = float(np.clip(spd_total, 0, self.pref_speed))
            f_hd = hd_total

        # =========================================================
        # v, w 转换
        # RL 激活时（alpha > 0）放松转弯减速和安全减速，
        # 让机器人能带速转向绕过障碍物，而不是停在墙前面
        # =========================================================
        herr = wrap(f_hd - self.cyaw)
        w = 3.0 * herr
        w = 0.7 * w + 0.3 * self.prev_w
        self.prev_w = w

        v = f_spd * (self.max_v / self.pref_speed)
        ae = abs(herr)

        if alpha > 0.1:
            # RL 激活：放松转弯减速，让机器人边走边转绕过障碍物
            if ae > 1.2:
                v *= 0.4    # 仍然减一些，但不压到 0.2
            elif ae > 0.6:
                v *= 0.6
            # 保持最低速度，防止卡死
            v = max(v, 0.08)
        else:
            # 纯 APF 模式：正常转弯减速
            if ae > 0.8:
                v *= 0.2
            elif ae > 0.4:
                v *= 0.4

        # 接近目标时保持最低速度
        if dist < 1.5:
            v = max(v, 0.10)

        # 前方障碍物安全减速
        if alpha > 0.1:
            # RL 激活时：放松安全减速，让 RL 的转向指令能执行
            # RL 已经在修正 heading 让机器人绕过去，不需要急刹
            if front_min < 0.15:
                v = min(v, 0.03)   # 极近才急刹（防撞底线）
            elif front_min < 0.3:
                v = min(v, 0.10)   # 原来是 0.03，现在放松到 0.10
        else:
            # 纯 APF 模式：正常安全减速
            if front_min < 0.3:
                v = min(v, 0.03)
            elif front_min < 0.5:
                v = min(v, 0.10)

        v = float(np.clip(v, 0.0, self.max_v))
        w = float(np.clip(w, -self.max_w, self.max_w))
        self.cmd(v, w)

        esc_str = f' ESC:{self.stuck_count}' if escape_active else ''
        ped_str = '🚶' if self.ped_received else '⏳'
        # ✅ v19: 日志标记 GT 表示使用 Ground Truth 坐标
        self.get_logger().info(
            f'GT:({self.cx:.1f},{self.cy:.1f}) 距:{dist:.1f}m | '
            f'APF:s={spd_apf:.2f},h={math.degrees(hd_apf):.0f}° | {rl_info} | '
            f'fin:s={f_spd:.2f} | v={v:.2f},w={w:.2f} | 前:{front_min:.1f}m{esc_str} {ped_str}'
        )


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ResidualRLDeployNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()