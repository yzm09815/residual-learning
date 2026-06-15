#!/usr/bin/env python3
"""
v21b_v3: A* 全局规划 + APF+SAC 残差局部避障（三段式遭遇版）
全局+局部融合
vs v21b 改动：
  ✅ 起点 (-8,-8)，终点 (8,8) — 恢复原来的长路径
  ✅ 5个行人分布在三个遭遇区：
     Zone1（起点附近）: P1 下方缺口 — 1个行人
     Zone2（中部）: P2+P3+P4 两缺口之间 — 3个行人密集
     Zone3（终点附近）: P5 右上区域 — 1个行人
  ✅ 流程: APF→RL→APF→RL→APF→APF→RL→终点
  ✅ 行人巡逻范围拉开，缺口不会被堵死
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
import numpy as np
import math
import subprocess
import re
import threading

from stable_baselines3 import SAC
from global_planner import GlobalPlanner


def wrap(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def wrap_train(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class ResidualRLDeployNode(Node):
    def __init__(self):
        super().__init__('residual_rl_deploy_node')

        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(Float32MultiArray, '/pedestrian_states', self.ped_cb, 10)

        # ============================================
        #  基本参数 — 起终点恢复原来的长路径
        # ============================================
        self.start_x = -8.0
        self.start_y = -8.0
        self.target_x = 8.0
        self.target_y = 8.0
        self.pref_speed = 1.0
        self.radius = 0.3

        # APF 参数
        self.k_att = 1.0
        self.k_rep = 0.5
        self.d0_ped = 3.0
        self.d0_wall = 1.0
        self.k_rep_wall = 0.5
        self.max_wall_force = 0.5

        self.max_other = 3
        self.num_ped = 5
        self.max_v = 0.26
        self.max_w = 2.0

        self.cx = self.start_x
        self.cy = self.start_y
        self.cyaw = 0.0
        self.cvx = 0.0
        self.cvy = 0.0
        self.prev_w = 0.0

        # Ground Truth
        self.gt_ready = False
        self._gt_lock = threading.Lock()
        self._start_gz_listener()

        # 卡住检测与重规划
        self.stuck_count = 0
        self.last_dist = 999.0
        self.stuck_threshold = 300
        self.replan_count = 0
        self.max_replans = 5
        self.start_time = None

        # ✅ 5 个行人状态 — 三段式遭遇布局
        # [x, y, vx, vy, speed, goal_x, goal_y, pref_speed, radius]
        self.ped_states = [
            # Zone1: P1 下方缺口区巡逻（起点附近遭遇）
            [-1.0, -3.0, 0.0, 0.0, 0.5, 2.0, -2.8, 0.5, 0.3],
            # Zone2: P2 两缺口之间对角穿越（中部遭遇）
            [0.0, -0.5, 0.0, 0.0, 0.7, 3.0, 1.0, 0.7, 0.3],
            # Zone2: P3 中央缺口右半段巡逻（中部遭遇）
            [3.0, 2.0, 0.0, 0.0, 0.45, 3.8, 1.8, 0.45, 0.3],
            # Zone2: P4 两缺口间纵向穿越（中部遭遇）
            [2.0, -1.0, 0.0, 0.0, 0.65, 2.5, 1.5, 0.65, 0.3],
            # Zone3: P5 右上区域巡逻（终点附近遭遇）
            [6.5, 7.0, 0.0, 0.0, 0.6, 8.0, 6.0, 0.6, 0.3],
        ]
        self.ped_received = False

        self._diag_cnt = 0
        self._diag_interval = 20

        # A* 全局规划器
        self.global_planner = GlobalPlanner(resolution=0.2, robot_radius=self.radius)
        self.waypoints = []
        self.current_wp_idx = 0
        self.current_wp_x = self.target_x
        self.current_wp_y = self.target_y
        self.wp_reach_threshold = 0.8

        self._do_global_plan()

        # RL 模型
        self.use_rl = False
        model_path = '/mnt/c/Users/admin/Desktop/训练模型/wall/2_动态残差_30%0.7_sac_1.3_1.3_12x12wall/sac_progressive_v8b_1500000_steps'
        try:
            self.model = SAC.load(model_path)
            self.get_logger().info(f'🧠 SAC 加载成功')
            self.use_rl = True
        except Exception as e:
            self.get_logger().error(f'❌ 模型加载失败: {e}')

        self.get_logger().info(f'🎯 起点:({self.start_x},{self.start_y}) → 终点:({self.target_x},{self.target_y})')
        self.get_logger().info(f'🗺️ v21b_v3: A* + APF + SAC | 三段式遭遇 | {self.num_ped}行人')
        self.get_logger().info(f'📍 Waypoints: {len(self.waypoints)} 个')
        self.get_logger().info(f'🚶 Zone1:P1(缺口) Zone2:P2+P3+P4(中部) Zone3:P5(终点)')

    # =========================================================
    #  全局规划
    # =========================================================
    def _do_global_plan(self):
        start = (self.cx, self.cy)
        goal = (self.target_x, self.target_y)
        self.get_logger().info(f'🗺️ A* 规划: ({start[0]:.1f},{start[1]:.1f}) → ({goal[0]:.1f},{goal[1]:.1f})')
        waypoints = self.global_planner.plan(start, goal, min_spacing=1.5)
        if waypoints and len(waypoints) >= 2:
            self.waypoints = waypoints
            self.current_wp_idx = 1
            self._update_current_wp()
            self.get_logger().info(f'✅ 规划成功！{len(self.waypoints)} 个 waypoints:')
            for i, wp in enumerate(self.waypoints):
                marker = "→" if i == self.current_wp_idx else " "
                self.get_logger().info(f'  {marker} wp{i}: ({wp[0]:.1f}, {wp[1]:.1f})')
        else:
            self.get_logger().warn('⚠️ A* 规划失败，使用直线导航')
            self.waypoints = [(self.target_x, self.target_y)]
            self.current_wp_idx = 0
            self._update_current_wp()

    def _do_replan_with_pedestrians(self):
        if self.replan_count >= self.max_replans:
            return
        ped_positions = [(self.ped_states[i][0], self.ped_states[i][1]) for i in range(self.num_ped)]
        start = (self.cx, self.cy)
        goal = (self.target_x, self.target_y)
        self.get_logger().info(f'🔄 动态重规划（第{self.replan_count+1}次）')
        waypoints = self.global_planner.replan_with_pedestrians(start, goal, ped_positions)
        if waypoints and len(waypoints) >= 2:
            self.waypoints = waypoints
            self.current_wp_idx = 1
            self._update_current_wp()
            self.replan_count += 1
            self.stuck_count = 0
            self.get_logger().info(f'✅ 重规划成功！新路径 {len(self.waypoints)} 个 waypoints')
        else:
            self.get_logger().warn('⚠️ 重规划失败')

    def _update_current_wp(self):
        if self.current_wp_idx < len(self.waypoints):
            wp = self.waypoints[self.current_wp_idx]
            self.current_wp_x = wp[0]
            self.current_wp_y = wp[1]
        else:
            self.current_wp_x = self.target_x
            self.current_wp_y = self.target_y

    def _advance_waypoint(self):
        dist_to_wp = math.hypot(self.cx - self.current_wp_x, self.cy - self.current_wp_y)
        if dist_to_wp < self.wp_reach_threshold:
            old_idx = self.current_wp_idx
            self.current_wp_idx += 1
            self._update_current_wp()
            if self.current_wp_idx < len(self.waypoints):
                self.get_logger().info(
                    f'📍 到达 wp{old_idx}! → wp{self.current_wp_idx}: '
                    f'({self.current_wp_x:.1f}, {self.current_wp_y:.1f})')
            else:
                self.get_logger().info(f'📍 所有 waypoints 已通过')
            self.stuck_count = 0
            return True
        return False

    # =========================================================
    #  Ground Truth
    # =========================================================
    def _start_gz_listener(self):
        self._gz_thread = threading.Thread(target=self._gz_listener_loop, daemon=True)
        self._gz_thread.start()

    def _gz_listener_loop(self):
        cmd = ['gz', 'topic', '-e', '-t',
               '/world/complex_nav_world/dynamic_pose/info']
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, bufsize=1)
            buffer_lines = []
            for line in proc.stdout:
                stripped = line.strip()
                if stripped == '' or stripped == '---':
                    if buffer_lines:
                        self._parse_frame('\n'.join(buffer_lines))
                        buffer_lines = []
                    continue
                buffer_lines.append(stripped)
        except Exception as e:
            self.get_logger().error(f'🛰️ gz listener 异常: {e}')

    def _parse_frame(self, frame_text):
        lines = frame_text.split('\n')
        waffle_idx = -1
        for i, line in enumerate(lines):
            if '"waffle"' in line:
                waffle_idx = i
                break
        if waffle_idx < 0:
            return
        block = '\n'.join(lines[max(0, waffle_idx-5):min(len(lines), waffle_idx+15)])
        pos_x = self._extract_field(block, 'position', 'x')
        pos_y = self._extract_field(block, 'position', 'y')
        ori_x = self._extract_field(block, 'orientation', 'x')
        ori_y = self._extract_field(block, 'orientation', 'y')
        ori_z = self._extract_field(block, 'orientation', 'z')
        ori_w = self._extract_field(block, 'orientation', 'w')
        if pos_x is not None and pos_y is not None:
            gyaw = self.cyaw
            if all(v is not None for v in [ori_x, ori_y, ori_z, ori_w]):
                gyaw = math.atan2(2.0*(ori_w*ori_z+ori_x*ori_y),
                                  1.0-2.0*(ori_y*ori_y+ori_z*ori_z))
            with self._gt_lock:
                self.cx = pos_x
                self.cy = pos_y
                self.cyaw = gyaw
                if not self.gt_ready:
                    self.gt_ready = True
                    self.get_logger().info(f'🛰️ 首次 GT: ({pos_x:.3f}, {pos_y:.3f})')

    def _extract_field(self, block, section, field):
        sec_match = re.search(rf'{section}\s*\{{', block)
        if not sec_match:
            return None
        start = sec_match.end()
        brace_count = 1
        end = start
        while end < len(block) and brace_count > 0:
            if block[end] == '{': brace_count += 1
            elif block[end] == '}': brace_count -= 1
            end += 1
        section_content = block[start:end]
        field_match = re.search(rf'{field}:\s*([-\d.eE+]+)', section_content)
        if field_match:
            try: return float(field_match.group(1))
            except ValueError: return None
        return None

    # =========================================================
    #  基础方法
    # =========================================================
    def cmd(self, v, w):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_footprint'
        m.twist.linear.x = float(v)
        m.twist.angular.z = float(w)
        self.pub.publish(m)

    def odom_cb(self, msg):
        self.cvx = msg.twist.twist.linear.x
        self.cvy = msg.twist.twist.linear.y

    def ped_cb(self, msg):
        expected = self.num_ped * 9
        if len(msg.data) >= expected:
            for i in range(self.num_ped):
                base = i * 9
                self.ped_states[i] = [float(msg.data[base + j]) for j in range(9)]
            if not self.ped_received:
                self.ped_received = True
                self.get_logger().info(f'🚶 收到 {self.num_ped} 个行人数据！')

    # =========================================================
    #  APF — 对所有 5 个行人算斥力
    # =========================================================
    def calculate_apf_force_aligned(self):
        pos = np.array([self.cx, self.cy])
        goal = np.array([self.current_wp_x, self.current_wp_y])
        goal_vec = goal - pos
        dist_goal = np.linalg.norm(goal_vec)
        if dist_goal > 0:
            f_att = self.k_att * (goal_vec / dist_goal) * self.pref_speed
        else:
            f_att = np.zeros(2)

        f_rep = np.zeros(2)
        for i in range(self.num_ped):
            ps = self.ped_states[i]
            other_pos = np.array([ps[0], ps[1]])
            other_radius = ps[8]
            diff = pos - other_pos
            dist = np.linalg.norm(diff)
            dist_edge = dist - self.radius - other_radius
            if dist_edge <= 0.05:
                dist_edge = 0.05
            if dist_edge < self.d0_ped:
                mag = self.k_rep * (1.0/dist_edge - 1.0/self.d0_ped) * (1.0/(dist_edge**2))
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
                for j in range(self.num_ped):
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

        wall_force_mag = np.linalg.norm(f_rep_wall)
        if wall_force_mag > self.max_wall_force:
            f_rep_wall = f_rep_wall / wall_force_mag * self.max_wall_force
        return f_rep_wall, front_min

    # =========================================================
    #  获取最近 3 个行人（给 RL obs 用）
    # =========================================================
    def _get_closest_3_ped_indices(self):
        dists = []
        for i in range(self.num_ped):
            ps = self.ped_states[i]
            d = math.hypot(ps[0] - self.cx, ps[1] - self.cy)
            dists.append((i, d))
        dists.sort(key=lambda x: x[1])
        return [idx for idx, _ in dists[:3]]

    def _get_min_ped_dist(self):
        min_d = float('inf')
        for i in range(self.num_ped):
            ps = self.ped_states[i]
            d = math.hypot(ps[0] - self.cx, ps[1] - self.cy) - self.radius - ps[8]
            min_d = min(min_d, d)
        return min_d

    # =========================================================
    #  OBS 构建 — 只用最近 3 个行人，保持 108 维
    # =========================================================
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
        ref_prll_angle = np.arctan2(ref_prll[1], ref_prll[0])
        return wrap(heading_global - ref_prll_angle)

    def _compute_other_agent_obs(self, host_pos, host_radius, host_ref_prll, host_ref_orth,
                                  other_pos, other_vel, other_radius):
        rel_pos = other_pos - host_pos
        p_prll = np.dot(rel_pos, host_ref_prll)
        p_orth = np.dot(rel_pos, host_ref_orth)
        v_prll = np.dot(other_vel, host_ref_prll)
        v_orth = np.dot(other_vel, host_ref_orth)
        dist_center = np.linalg.norm(rel_pos)
        combined_radius = host_radius + other_radius
        dist_edge = dist_center - combined_radius
        return np.array([p_prll, p_orth, v_prll, v_orth,
                         other_radius, combined_radius, dist_edge], dtype=np.float32)

    def build_single_agent_obs_alphabetical(self, agent_pos, agent_goal, agent_heading,
                                             agent_vel, agent_pref_speed, agent_radius,
                                             is_learning_val, other_agents_info):
        pos = np.array(agent_pos)
        goal = np.array(agent_goal)
        ref_prll, ref_orth, dist_to_goal = self._get_ref(pos, goal)
        heading_ego = self._get_heading_ego(agent_heading, ref_prll)

        other_states = np.zeros((self.max_other, 7), dtype=np.float32)
        num_observed = 0
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
                np.array(o_pos), np.array(o_vel), o_rad)
            other_states[rank] = obs_7
            num_observed += 1

        obs = np.concatenate([
            np.array([dist_to_goal], dtype=np.float32),
            np.array([heading_ego], dtype=np.float32),
            np.array([is_learning_val], dtype=np.float32),
            np.array([float(num_observed)], dtype=np.float32),
            other_states.flatten(),
            np.array([agent_pref_speed], dtype=np.float32),
            np.array([agent_radius], dtype=np.float32),
        ])
        return obs

    def build_full_state(self):
        robot_pos = [self.cx, self.cy]
        robot_goal = [self.current_wp_x, self.current_wp_y]
        robot_heading = self.cyaw
        robot_speed = math.hypot(self.cvx, self.cvy)
        robot_vel = [robot_speed * math.cos(self.cyaw),
                     robot_speed * math.sin(self.cyaw)]

        closest_indices = self._get_closest_3_ped_indices()

        all_agents = []
        all_agents.append({
            'pos': robot_pos, 'goal': robot_goal, 'heading': robot_heading,
            'vel': robot_vel, 'pref_speed': self.pref_speed,
            'radius': self.radius, 'is_learning': 0.0,
        })

        for i in closest_indices:
            ps = self.ped_states[i]
            ped_vel = [ps[2], ps[3]]
            if abs(ps[2]) + abs(ps[3]) > 0.01:
                ped_heading = math.atan2(ps[3], ps[2])
            else:
                goal_dir = np.array([ps[5], ps[6]]) - np.array([ps[0], ps[1]])
                ped_heading = math.atan2(goal_dir[1], goal_dir[0])
            all_agents.append({
                'pos': [ps[0], ps[1]], 'goal': [ps[5], ps[6]],
                'heading': ped_heading, 'vel': ped_vel,
                'pref_speed': ps[7], 'radius': ps[8], 'is_learning': 0.0,
            })

        all_obs = []
        for agent_idx in range(4):
            agent = all_agents[agent_idx]
            others = []
            for j in range(4):
                if j == agent_idx:
                    continue
                other = all_agents[j]
                others.append((other['pos'], other['vel'], other['radius']))
            obs_27 = self.build_single_agent_obs_alphabetical(
                agent['pos'], agent['goal'], agent['heading'],
                agent['vel'], agent['pref_speed'], agent['radius'],
                agent['is_learning'], others)
            all_obs.append(obs_27)

        return np.concatenate(all_obs)

    def _run_diagnostics(self, state, apf_speed, apf_heading):
        self._diag_cnt += 1
        if self._diag_cnt % self._diag_interval != 0:
            return
        a0 = state[:27]
        self.get_logger().info(f'  [诊断] wp{self.current_wp_idx}/{len(self.waypoints)} '
                               f'd={a0[0]:.2f} heading={math.degrees(a0[1]):.0f}° '
                               f'APF: s={apf_speed:.2f} h={math.degrees(apf_heading):.0f}°')

    # =========================================================
    #  主控制循环
    # =========================================================
    def scan_cb(self, msg):
        if not self.gt_ready or not self.ped_received:
            self.cmd(0.0, 0.0)
            return

        for i in range(self.num_ped):
            ps = self.ped_states[i]
            d = math.hypot(ps[0] - self.cx, ps[1] - self.cy)
            if d < (self.radius + ps[8]):
                self.get_logger().info(f'💥 碰撞行人 {i+1}！停止')
                self.cmd(0.0, 0.0)
                return

        dist_final = math.hypot(self.target_x - self.cx, self.target_y - self.cy)
        if dist_final < 0.3:
            elapsed = 0.0
            if self.start_time is not None:
                elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            self.get_logger().info(
                f'🎉 到达！GT:({self.cx:.2f},{self.cy:.2f}) | ⏱️ 耗时: {elapsed:.1f}秒')
            self.cmd(0.0, 0.0)
            return

        self._advance_waypoint()

        dist_to_wp = math.hypot(self.current_wp_x - self.cx, self.current_wp_y - self.cy)
        if abs(dist_to_wp - self.last_dist) < 0.03:
            self.stuck_count += 1
        else:
            self.stuck_count = max(0, self.stuck_count - 1)
        self.last_dist = dist_to_wp

        if self.stuck_count > self.stuck_threshold:
            self.get_logger().warn(f'⚠️ 卡住 {self.stuck_count} 步，触发重规划')
            self._do_replan_with_pedestrians()

        f_apf = self.calculate_apf_force_aligned()
        f_wall, front_min = self.calculate_wall_repulsion(msg)
        f_apf_with_wall = f_apf + f_wall
        spd_apf = np.linalg.norm(f_apf_with_wall)
        hd_apf = math.atan2(f_apf_with_wall[1], f_apf_with_wall[0])

        rl_info = 'APF'
        alpha = 0.0

        if self.use_rl and self.ped_received:
            state = self.build_full_state()
            self._run_diagnostics(state, spd_apf, hd_apf)
            action, _ = self.model.predict(state, deterministic=True)

            ds_raw = action[0] * 2.0 * self.pref_speed
            dh_raw = action[1] * 2.0

            min_ped_dist = self._get_min_ped_dist()

            alpha_max = 0.8
            dist_far = 1.5
            dist_near = 0.6
            if min_ped_dist >= dist_far:
                alpha = 0.0
            elif min_ped_dist <= dist_near:
                alpha = alpha_max
            else:
                alpha = alpha_max * (dist_far - min_ped_dist) / (dist_far - dist_near)

            if front_min < 0.8:
                alpha = min(alpha, 0.2)

            dh_clipped = float(np.clip(dh_raw, -1.2, 1.2))
            ds_clipped = float(np.clip(ds_raw, -0.5*self.pref_speed, 1.5*self.pref_speed))

            f_spd = float(np.clip(spd_apf + alpha*ds_clipped, 0.10*self.pref_speed, self.pref_speed))
            f_hd = wrap_train(hd_apf + alpha*dh_clipped)

            if alpha > 0.01:
                rl_info = f'AI:α={alpha:.2f} ped={min_ped_dist:.1f}'
        else:
            f_spd = float(np.clip(spd_apf, 0, self.pref_speed))
            f_hd = hd_apf

        herr = wrap(f_hd - self.cyaw)
        w = 3.0 * herr
        w = 0.7 * w + 0.3 * self.prev_w
        self.prev_w = w

        v = f_spd * (self.max_v / self.pref_speed)
        ae = abs(herr)

        if alpha > 0.1:
            if ae > 1.2: v *= 0.4
            elif ae > 0.6: v *= 0.6
            v = max(v, 0.08)
        else:
            if ae > 0.8: v *= 0.2
            elif ae > 0.4: v *= 0.4

        if dist_to_wp < 1.5:
            v = max(v, 0.10)

        if front_min < 0.3:
            v = min(v, 0.03)
        elif front_min < 0.5:
            v = min(v, 0.10)

        if self.start_time is None and (v > 0.01 or abs(w) > 0.01):
            self.start_time = self.get_clock().now()
            self.get_logger().info('⏱️ 计时开始！')

        v = float(np.clip(v, 0.0, self.max_v))
        w = float(np.clip(w, -self.max_w, self.max_w))
        self.cmd(v, w)

        wp_str = f'wp{self.current_wp_idx}/{len(self.waypoints)}'
        self.get_logger().info(
            f'GT:({self.cx:.1f},{self.cy:.1f}) {wp_str}→({self.current_wp_x:.1f},{self.current_wp_y:.1f}) '
            f'd={dist_to_wp:.1f}m | {rl_info} | v={v:.2f},w={w:.2f} | 前:{front_min:.1f}m 🚶')


def main(args=None):
    rclpy.init(args=args)
    node = ResidualRLDeployNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()