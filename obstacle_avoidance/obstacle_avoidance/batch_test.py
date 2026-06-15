#!/usr/bin/env python3
"""
batch_test.py — 复杂地图大规模批量测试（500 回合）

三种模式（SAC_MODE 控制）：
  - "apf"      : 纯 APF 基线（不调用 SAC）
  - "residual" : APF + SAC 残差（原方案）
  - "pure_sac" : 纯 SAC（接近障碍/行人时完全用 SAC 输出，远离时仍用 APF 跟 A*）

注意：USE_SAC 已废弃，改用 SAC_MODE。为兼容旧脚本仍保留 USE_SAC。
"""

import os
import sys
import math
import time
import numpy as np

sys.path.insert(0, '/home/claude')

from global_planner import GlobalPlanner, GridMap


# ============================================================
# 参数
# ============================================================

START_XY = (-8.0, -8.0)
GOAL_XY = (8.0, 8.0)

DT = 0.1
MAX_STEPS = 1500
TEST_EPISODES = 500

ROBOT_RADIUS = 0.3
PREF_SPEED = 1.0
MAX_V = 0.26
MAX_W = 2.0

K_ATT = 1.0
K_REP = 0.5
D0_PED = 3.0
D0_WALL = 1.0
K_REP_WALL = 0.5
MAX_WALL_FORCE = 0.5

WP_REACH_THRESHOLD = 0.8
GOAL_REACH_THRESHOLD = 0.5

STUCK_THRESHOLD = 300
STUCK_DELTA = 0.03
MAX_REPLANS = 5

ALPHA_MAX = 0.8
DIST_FAR = 1.5
DIST_NEAR = 0.6

# pure_sac 专用阈值（比 residual 大很多）
# 训练环境里 agent 在半径 2~4m 的圆上，从 ~4m 就开始有行人交互
# SAC 需要足够距离来做出避让决策
DIST_FAR_PURE = 4.0     # 行人 ≥ 4m 时 APF 沿 waypoint 导航
DIST_NEAR_PURE = 1.5     # 行人 ≤ 1.5m 时 SAC 完全接管
WALL_HARD_CUTOFF = 0.4
FRONT_BLOCK_THRESH = 0.6
WALL_SOFT_CAP = 0.2

DH_CLIP = 1.2
DS_CLIP_MIN = -0.5
DS_CLIP_MAX = 1.5

LIDAR_NUM_RAYS = 360
LIDAR_MAX_RANGE = 3.5
LIDAR_ANGLE_MIN = -math.pi
LIDAR_ANGLE_INC = 2 * math.pi / LIDAR_NUM_RAYS

MODEL_PATH = "/mnt/c/Users/admin/Desktop/训练模型/wall/2_动态残差_30%0.7_sac_1.3_1.3_12x12wall/sac_progressive_v8b_1500000_steps"

# 纯 SAC 独立训练的模型路径（训练脚本输出 sac_pure_v7_walls_final.zip 或 checkpoint）
# 改成你实际的纯 SAC 模型位置
MODEL_PATH_PURE_SAC = "/mnt/c/Users/admin/Desktop/训练模型/wall/9_pure_30%0.7_sac_12x12wall/sac_pure_v7_walls_1500000_steps"

PED_SPEED_JITTER = 0.20
PED_SPEED_MULTIPLIER = 1.0

PED_COLLISION_FACTOR = 0.85
WALL_COLLISION_FACTOR = 1.0

MAX_OTHER = 3
OBS_DIM = 108

# ============================================================
# 模式开关（三选一）：
#   "apf"      = 纯 APF 基线
#   "residual" = APF + SAC 残差（原方案）
#   "pure_sac" = 纯 SAC（接近障碍时 SAC 完全接管，APF 仅用于远距离跟 A*）
# ============================================================
SAC_MODE = "apf"

# 兼容旧开关：如果 USE_SAC 在外部被设了，就映射过来
USE_SAC = (SAC_MODE != "apf")

# 纯 SAC 模式下 SAC 输出的解释方式：
#   action[0] ∈ [-1,1] → speed = (action[0]+1)/2 × pref_speed   (映射到 [0, pref])
#   action[1] ∈ [-1,1] → heading = cyaw + action[1] × HEADING_RANGE
PURE_SAC_HEADING_RANGE = math.pi   # ±180° 转向范围

# pure_sac 振荡兜底：连续两步 SAC heading 输出突变 > OSCILLATION_THRESH 时，
# 强制 APF 接管 OSCILLATION_HOLD_STEPS 步
OSCILLATION_THRESH = math.radians(90.0)
OSCILLATION_HOLD_STEPS = 5

# pure_sac obs 修正：训练地图 12×12，dist_to_goal 典型 4~10
# 测试地图 20×20 + A* waypoint 间距不固定，dist 可能远超训练分布
# 截断到训练最大值，防止 OOD
DIST_TO_GOAL_CLIP = 10.0

# DEBUG：仅在 pure_sac 模式生效，打印行人接近时 SAC 的实际输出，便于定位"为什么不躲"
# 收官 500 回合时设 False（关闭打印，加速）
DEBUG_PURE_SAC = False


# ============================================================
# 工具函数
# ============================================================

def wrap(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def wrap_train(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ============================================================
# 行人
# ============================================================

PEDESTRIAN_SPECS = [
    ('P1', 0.5, 0.3, (-1.0, -3.0), [(-2.5, -2.8), (0.5, -2.8), (0.5, -3.2), (-2.5, -3.2)]),
    ('P2', 0.7, 0.3, ( 0.0, -0.5), [(-2.0, -2.0), ( 3.0,  1.0), ( 3.5, -1.0), (-1.0, -2.5)]),
    ('P3', 0.45, 0.3, ( 3.0,  2.0), [( 2.0,  1.8), ( 3.8,  1.8), ( 3.8,  2.2), ( 2.0,  2.2)]),
    ('P4', 0.65, 0.3, ( 2.0, -1.0), [( 2.0, -2.0), ( 2.0,  1.5), ( 2.5,  1.5), ( 2.5, -2.0)]),
    ('P5', 0.6, 0.3, ( 6.5,  7.0), [( 5.0,  7.0), ( 8.0,  7.0), ( 8.0,  6.0), ( 5.0,  6.0)]),
]


class Pedestrian:
    def __init__(self, spec, speed_jitter=0.0, speed_multiplier=1.0):
        name, base_speed, radius, init_pos, waypoints = spec
        self.name = name
        self.radius = radius
        jitter = 1.0 + np.random.uniform(-speed_jitter, speed_jitter)
        self.pref_speed = base_speed * speed_multiplier * jitter
        self.pos = np.array(init_pos, dtype=np.float64)
        self.vel = np.zeros(2)
        self.waypoints = list(waypoints)
        self.current_wp = 0

    def step(self, dt):
        wp = np.array(self.waypoints[self.current_wp])
        diff = wp - self.pos
        dist = np.linalg.norm(diff)
        if dist < 0.3:
            self.current_wp = (self.current_wp + 1) % len(self.waypoints)
            wp = np.array(self.waypoints[self.current_wp])
            diff = wp - self.pos
            dist = np.linalg.norm(diff)
        if dist > 1e-3:
            direction = diff / dist
            self.vel = direction * self.pref_speed
        else:
            self.vel = np.zeros(2)
        self.pos = self.pos + self.vel * dt


# ============================================================
# 颗粒墙 / 碰撞地图
# ============================================================

def build_wall_points(grid_map, subsample=1):
    pts = []
    for gy in range(0, grid_map.height, subsample):
        for gx in range(0, grid_map.width, subsample):
            if grid_map.grid[gy, gx] == 1:
                wx = grid_map.grid_to_world_x(gx)
                wy = grid_map.grid_to_world_y(gy)
                pts.append((wx, wy))
    return np.array(pts, dtype=np.float64)


def build_collision_grid(resolution=0.05):
    walls = [
        (0.0, 10.0, 20.4, 0.2, 0),
        (0.0, -10.0, 20.4, 0.2, 0),
        (-10.0, 0.0, 20.4, 0.2, 90),
        (10.0, 0.0, 20.4, 0.2, 90),
        (-3.5, 2.0, 7.0, 0.2, 0),
        (6.5, 2.0, 5.0, 0.2, 0),
        (-5.5, -3.0, 5.0, 0.2, 0),
        (5.5, -3.0, 5.0, 0.2, 0),
        (-7.5, -6.5, 3.0, 0.2, 90),
        (-6.0, -8.0, 3.0, 0.2, 0),
        (-4.5, -6.5, 3.0, 0.2, 90),
        (0.0, -6.5, 4.0, 0.2, 90),
        (7.0, 5.5, 5.0, 0.2, 90),
        (3.5, 0.0, 0.8, 0.8, 0),
        (-3.0, -1.0, 0.8, 0.8, 0),
    ]
    x_min, x_max = -10.0, 10.0
    y_min, y_max = -10.0, 10.0
    width = int((x_max - x_min) / resolution)
    height = int((y_max - y_min) / resolution)
    grid = np.zeros((height, width), dtype=np.uint8)

    def w2gx(wx): return int((wx - x_min) / resolution)
    def w2gy(wy): return int((wy - y_min) / resolution)
    def g2wx(gx): return x_min + (gx + 0.5) * resolution
    def g2wy(gy): return y_min + (gy + 0.5) * resolution

    for cx, cy, length, thickness, angle_deg in walls:
        ar = math.radians(angle_deg)
        ca, sa = math.cos(ar), math.sin(ar)
        half_l = length / 2.0
        half_t = thickness / 2.0
        corners = [(cx + ca*dx - sa*dy, cy + sa*dx + ca*dy)
                   for dx in [-half_l, half_l] for dy in [-half_t, half_t]]
        bx_min = min(c[0] for c in corners) - resolution
        bx_max = max(c[0] for c in corners) + resolution
        by_min = min(c[1] for c in corners) - resolution
        by_max = max(c[1] for c in corners) + resolution
        for gy in range(max(0, w2gy(by_min)), min(height, w2gy(by_max)+1)):
            for gx in range(max(0, w2gx(bx_min)), min(width, w2gx(bx_max)+1)):
                wx, wy = g2wx(gx), g2wy(gy)
                dx, dy = wx - cx, wy - cy
                lx = ca*dx + sa*dy
                ly = -sa*dx + ca*dy
                if abs(lx) <= half_l and abs(ly) <= half_t:
                    grid[gy, gx] = 1

    class RawCollisionGrid: pass
    g = RawCollisionGrid()
    g.grid = grid
    g.width = width
    g.height = height
    g.resolution = resolution
    g.x_min, g.y_min = x_min, y_min
    g.world_to_grid = lambda wx, wy: (int((wx-x_min)/resolution), int((wy-y_min)/resolution))
    g.grid_to_world_x = lambda gx: x_min + (gx + 0.5) * resolution
    g.grid_to_world_y = lambda gy: y_min + (gy + 0.5) * resolution
    return g


def is_robot_collided_with_wall(collision_grid, x, y, robot_radius):
    res = collision_grid.resolution
    half = res / 2.0
    r_search = robot_radius + res
    r_cells = int(r_search / res) + 1
    gx_c, gy_c = collision_grid.world_to_grid(x, y)
    r2 = robot_radius * robot_radius
    for dy in range(-r_cells, r_cells + 1):
        for dx in range(-r_cells, r_cells + 1):
            gx = gx_c + dx
            gy = gy_c + dy
            if 0 <= gx < collision_grid.width and 0 <= gy < collision_grid.height:
                if collision_grid.grid[gy, gx] == 1:
                    cx = collision_grid.grid_to_world_x(gx)
                    cy = collision_grid.grid_to_world_y(gy)
                    dx_c = max(abs(x - cx) - half, 0.0)
                    dy_c = max(abs(y - cy) - half, 0.0)
                    if dx_c*dx_c + dy_c*dy_c < r2:
                        return True
    return False


# ============================================================
# 虚拟 LaserScan
# ============================================================

def virtual_laser_scan(robot_pos, robot_yaw, wall_pts):
    diffs = wall_pts - robot_pos
    dists = np.linalg.norm(diffs, axis=1)
    mask = dists < LIDAR_MAX_RANGE
    if not np.any(mask):
        return np.full(LIDAR_NUM_RAYS, LIDAR_MAX_RANGE, dtype=np.float32)
    diffs = diffs[mask]
    dists = dists[mask]
    angles_global = np.arctan2(diffs[:, 1], diffs[:, 0])
    angles_local = angles_global - robot_yaw
    angles_local = (angles_local + math.pi) % (2 * math.pi) - math.pi
    bins = ((angles_local - LIDAR_ANGLE_MIN) / LIDAR_ANGLE_INC).astype(np.int32)
    bins = np.clip(bins, 0, LIDAR_NUM_RAYS - 1)
    ranges = np.full(LIDAR_NUM_RAYS, LIDAR_MAX_RANGE, dtype=np.float32)
    np.minimum.at(ranges, bins, dists)
    return ranges


# ============================================================
# 控制器
# ============================================================

class DeployController:

    def __init__(self, model, planner, wall_pts, collision_grid):
        self.model = model
        self.planner = planner
        self.wall_pts = wall_pts
        self.collision_grid = collision_grid

        self.radius = ROBOT_RADIUS
        self.pref_speed = PREF_SPEED
        self.k_att = K_ATT
        self.k_rep = K_REP
        self.d0_ped = D0_PED
        self.d0_wall = D0_WALL
        self.k_rep_wall = K_REP_WALL
        self.max_wall_force = MAX_WALL_FORCE
        self.max_v = MAX_V
        self.max_w = MAX_W

        self.cx, self.cy = START_XY
        self.cyaw = math.atan2(GOAL_XY[1] - START_XY[1], GOAL_XY[0] - START_XY[0])
        self.cv = 0.0
        self.prev_w = 0.0

        self.waypoints = []
        self.current_wp_idx = 0
        self.current_wp_x = GOAL_XY[0]
        self.current_wp_y = GOAL_XY[1]

        self.stuck_count = 0
        self.last_dist = 999.0
        self.replan_count = 0

        # pure_sac 振荡检测状态
        self._last_sac_heading = None
        self._force_apf_remaining = 0
        self._sac_oscillating = False

        self._do_global_plan()

    def reset(self):
        self.cx, self.cy = START_XY
        self.cyaw = math.atan2(GOAL_XY[1] - START_XY[1], GOAL_XY[0] - START_XY[0])
        self.cv = 0.0
        self.prev_w = 0.0
        self.stuck_count = 0
        self.last_dist = 999.0
        self.replan_count = 0
        self._last_sac_heading = None
        self._force_apf_remaining = 0
        self._sac_oscillating = False
        self._do_global_plan()

    def _do_global_plan(self):
        wps = self.planner.plan(START_XY, GOAL_XY, min_spacing=1.5)
        if wps and len(wps) >= 2:
            self.waypoints = wps
            self.current_wp_idx = 1
            self.current_wp_x, self.current_wp_y = wps[1]
        else:
            self.waypoints = [GOAL_XY]
            self.current_wp_idx = 0
            self.current_wp_x, self.current_wp_y = GOAL_XY

    def _do_replan(self, peds):
        if self.replan_count >= MAX_REPLANS:
            return False
        ped_positions = [(p.pos[0], p.pos[1]) for p in peds]
        wps = self.planner.replan_with_pedestrians(
            (self.cx, self.cy), GOAL_XY, ped_positions)
        if wps and len(wps) >= 2:
            self.waypoints = wps
            self.current_wp_idx = 1
            self.current_wp_x, self.current_wp_y = wps[1]
            self.replan_count += 1
            self.stuck_count = 0
            return True
        return False

    def _advance_waypoint(self):
        d = math.hypot(self.cx - self.current_wp_x, self.cy - self.current_wp_y)
        if d < WP_REACH_THRESHOLD:
            self.current_wp_idx += 1
            if self.current_wp_idx < len(self.waypoints):
                self.current_wp_x, self.current_wp_y = self.waypoints[self.current_wp_idx]
            else:
                self.current_wp_x, self.current_wp_y = GOAL_XY
            self.stuck_count = 0

    def calc_apf(self, peds):
        pos = np.array([self.cx, self.cy])
        goal = np.array([self.current_wp_x, self.current_wp_y])
        gv = goal - pos
        d = np.linalg.norm(gv)
        f_att = (gv / d) * self.k_att * self.pref_speed if d > 0 else np.zeros(2)
        f_rep = np.zeros(2)
        for p in peds:
            diff = pos - p.pos
            dist = np.linalg.norm(diff)
            de = max(dist - self.radius - p.radius, 0.05)
            if de < self.d0_ped and dist > 1e-3:
                mag = self.k_rep * (1.0/de - 1.0/self.d0_ped) * (1.0/(de**2))
                f_rep += mag * (diff / dist)
        return f_att + f_rep

    def calc_wall_rep(self, ranges, peds):
        pos = np.array([self.cx, self.cy])
        f = np.zeros(2)
        front_min = 5.0
        for i, r in enumerate(ranges):
            if r <= 0 or r >= LIDAR_MAX_RANGE:
                continue
            sr = max(float(r), 0.15)
            al = LIDAR_ANGLE_MIN + i * LIDAR_ANGLE_INC
            if al > math.pi:
                al -= 2 * math.pi
            if abs(al) < 0.4:
                front_min = min(front_min, sr)
            if sr < self.d0_wall:
                ag = al + self.cyaw
                op = pos + sr * np.array([math.cos(ag), math.sin(ag)])
                is_ped = False
                for p in peds:
                    if np.linalg.norm(op - p.pos) < 0.5:
                        is_ped = True
                        break
                if is_ped:
                    continue
                diff = pos - op
                d = np.linalg.norm(diff)
                de = max(d - self.radius - 0.3, 0.05)
                if de < self.d0_wall and d > 1e-3:
                    mag = self.k_rep_wall * (1.0/de - 1.0/self.d0_wall) / (de**2)
                    mag = min(mag, 5.0)
                    f += mag * (diff / d)
        mag_wall = np.linalg.norm(f)
        if mag_wall > self.max_wall_force:
            f = f / mag_wall * self.max_wall_force
        return f, front_min

    def _ref_frame(self, pos, goal):
        gv = goal - pos
        d = np.linalg.norm(gv)
        if d > 1e-8:
            ref_prll = gv / d
        else:
            ref_prll = gv.copy()
        ref_orth = np.array([-ref_prll[1], ref_prll[0]])
        return ref_prll, ref_orth, d

    def _other_obs(self, host_pos, host_r, ref_p, ref_o, opos, ovel, orad):
        rel = opos - host_pos
        p_p = np.dot(rel, ref_p)
        p_o = np.dot(rel, ref_o)
        v_p = np.dot(ovel, ref_p)
        v_o = np.dot(ovel, ref_o)
        dc = np.linalg.norm(rel)
        cr = host_r + orad
        de = dc - cr
        return np.array([p_p, p_o, v_p, v_o, orad, cr, de], dtype=np.float32)

    def _single_agent_obs(self, apos, agoal, ahead, avel, apref, arad, is_learn, others):
        pos = np.array(apos)
        goal = np.array(agoal)
        ref_p, ref_o, dist = self._ref_frame(pos, goal)
        # ★ clamp dist_to_goal 到训练分布范围（训练地图 12×12，最大对角线 ~17，
        #   实际 spawn 范围 4~10；超出时截断，保持方向不变）
        dist = min(dist, DIST_TO_GOAL_CLIP)
        h_ego = wrap(ahead - math.atan2(ref_p[1], ref_p[0]))
        states = np.zeros((MAX_OTHER, 7), dtype=np.float32)
        dists = []
        for idx, (op, ov, orad) in enumerate(others):
            dists.append((idx, np.linalg.norm(np.array(op) - pos)))
        dists.sort(key=lambda x: x[1])
        n_obs = 0
        for rank, (idx, _) in enumerate(dists):
            if rank >= MAX_OTHER:
                break
            op, ov, orad = others[idx]
            states[rank] = self._other_obs(pos, arad, ref_p, ref_o,
                                           np.array(op), np.array(ov), orad)
            n_obs += 1
        return np.concatenate([
            np.array([dist], dtype=np.float32),
            np.array([h_ego], dtype=np.float32),
            np.array([is_learn], dtype=np.float32),
            np.array([float(n_obs)], dtype=np.float32),
            states.flatten(),
            np.array([apref], dtype=np.float32),
            np.array([arad], dtype=np.float32),
        ])

    def build_obs(self, peds, use_final_goal=False):
        dists = [(i, math.hypot(p.pos[0]-self.cx, p.pos[1]-self.cy)) for i, p in enumerate(peds)]
        dists.sort(key=lambda x: x[1])
        closest = [peds[i] for i, _ in dists[:3]]

        # 三模式统一用 A* waypoint 作为 goal（保证 dist_to_goal 在训练分布范围内）
        # 训练环境 dist_to_goal 典型 4~10；如果用最终目标(8,8)，dist=16~22 远超训练分布
        robot_goal = [self.current_wp_x, self.current_wp_y]

        robot_vel = [self.cv * math.cos(self.cyaw), self.cv * math.sin(self.cyaw)]
        # is_learning：pure_sac 训练时 agent0.is_learning=True(1.0)；residual/apf 沿用 0.0
        is_learn_robot = 1.0 if SAC_MODE == "pure_sac" else 0.0
        all_agents = [{
            'pos': [self.cx, self.cy], 'goal': robot_goal,
            'heading': self.cyaw, 'vel': robot_vel,
            'pref_speed': self.pref_speed, 'radius': self.radius, 'is_learning': is_learn_robot,
        }]
        for p in closest:
            if abs(p.vel[0]) + abs(p.vel[1]) > 0.01:
                ph = math.atan2(p.vel[1], p.vel[0])
            else:
                wp = np.array(p.waypoints[p.current_wp])
                gd = wp - p.pos
                ph = math.atan2(gd[1], gd[0])
            wp = p.waypoints[p.current_wp]
            all_agents.append({
                'pos': [p.pos[0], p.pos[1]], 'goal': [wp[0], wp[1]],
                'heading': ph, 'vel': [p.vel[0], p.vel[1]],
                'pref_speed': p.pref_speed, 'radius': p.radius, 'is_learning': 0.0,
            })

        obs_list = []
        for i in range(4):
            agent = all_agents[i]
            others = []
            for j in range(4):
                if j == i:
                    continue
                o = all_agents[j]
                others.append((o['pos'], o['vel'], o['radius']))
            obs_list.append(self._single_agent_obs(
                agent['pos'], agent['goal'], agent['heading'], agent['vel'],
                agent['pref_speed'], agent['radius'], agent['is_learning'], others))
        return np.concatenate(obs_list)

    def _min_ped_dist(self, peds):
        m = float('inf')
        for p in peds:
            d = math.hypot(p.pos[0]-self.cx, p.pos[1]-self.cy) - self.radius - p.radius
            if d < m: m = d
        return m

    # ----- 主控制步骤 -----
    def step(self, peds):
        # 行人碰撞
        for p in peds:
            d = math.hypot(p.pos[0]-self.cx, p.pos[1]-self.cy)
            if d < (self.radius + p.radius) * PED_COLLISION_FACTOR:
                return 'collision_ped', 0.0, 0.0

        # 终点
        if math.hypot(GOAL_XY[0]-self.cx, GOAL_XY[1]-self.cy) < GOAL_REACH_THRESHOLD:
            return 'reached', 0.0, 0.0

        self._advance_waypoint()

        # 卡住检测
        dwp = math.hypot(self.current_wp_x-self.cx, self.current_wp_y-self.cy)
        if abs(dwp - self.last_dist) < STUCK_DELTA:
            self.stuck_count += 1
        else:
            self.stuck_count = max(0, self.stuck_count - 1)
        self.last_dist = dwp
        if self.stuck_count > STUCK_THRESHOLD:
            self._do_replan(peds)

        # APF 永远算（pure_sac 模式下 α=0 时仍然用它跟 A*）
        f_apf = self.calc_apf(peds)
        ranges = virtual_laser_scan(np.array([self.cx, self.cy]), self.cyaw, self.wall_pts)
        f_wall, front_min = self.calc_wall_rep(ranges, peds)
        f_total = f_apf + f_wall
        spd_apf = np.linalg.norm(f_total)
        hd_apf = math.atan2(f_total[1], f_total[0])

        # SAC 推理（apf 模式不调用）
        # 关键：pure_sac 训练时 goal 字段是最终目标，不是 A* waypoint —— 必须对齐
        if SAC_MODE == "residual":
            obs = self.build_obs(peds)
            action, _ = self.model.predict(obs, deterministic=True)
            a0 = float(action[0])
            a1 = float(action[1])
        elif SAC_MODE == "pure_sac":
            obs = self.build_obs(peds)
            action, _ = self.model.predict(obs, deterministic=True)
            a0 = float(action[0])   # 训练 action[0] ∈ [0, 2.0]，直接是速度 m/s
            a1 = float(action[1])   # 训练 action[1] ∈ [-π, π]，直接是全局 heading

            # ===== 振荡检测：本步 SAC heading 与上一步突变 > OSCILLATION_THRESH 时
            # 触发 OSCILLATION_HOLD_STEPS 步的"APF 强制接管"窗口 =====
            if self._last_sac_heading is not None:
                jump = abs(wrap(a1 - self._last_sac_heading))
                if jump > OSCILLATION_THRESH:
                    self._force_apf_remaining = OSCILLATION_HOLD_STEPS
            self._last_sac_heading = a1
            if self._force_apf_remaining > 0:
                self._force_apf_remaining -= 1
                self._sac_oscillating = True
            else:
                self._sac_oscillating = False

            # ===== OBS DEBUG：只在进入 P3 附近时打印（cy>0 且 d_ped<2） =====
            if DEBUG_PURE_SAC and self.cy > 0 and self.cy < 3:
                obs_a0 = obs[:27]
                min_ped_dbg = self._min_ped_dist(peds)
                if min_ped_dbg < 2.0:
                    nearest = min(peds, key=lambda p: math.hypot(p.pos[0]-self.cx, p.pos[1]-self.cy))
                    bearing_to_ped = math.degrees(wrap(math.atan2(nearest.pos[1]-self.cy, nearest.pos[0]-self.cx) - self.cyaw))
                    osc_tag = " [OSC]" if self._sac_oscillating else ""
                    print(f"  [SAC] R=({self.cx:+.2f},{self.cy:+.2f}) yaw={math.degrees(self.cyaw):+5.0f}° "
                          f"d_ped={min_ped_dbg:.2f}m P{peds.index(nearest)+1} bear={bearing_to_ped:+5.0f}° "
                          f"| act=[{a0:.2f},Δhd={math.degrees(a1):+5.0f}°] α={alpha if hasattr(self,'_dbg_alpha') else 0:.2f}{osc_tag}")
                    print(f"    obs dist={obs_a0[0]:.2f} h_ego={obs_a0[1]:.2f} wp=({self.current_wp_x:+.1f},{self.current_wp_y:+.1f})")
        else:
            a0, a1 = 0.0, 0.0

        # 动态 α
        min_ped = self._min_ped_dist(peds)

        if SAC_MODE == "pure_sac":
            # 纯 SAC 模式：
            #   远距离 → APF 沿 A* waypoint 导航（SAC 没见过复杂地图障碍）
            #   近距离 → SAC 完全接管避让（SAC 擅长的）
            # 阈值比 residual 更大：训练时 agent 从 ~4m 就交互，给 SAC 足够反应时间
            if min_ped >= DIST_FAR_PURE:
                alpha = 0.0
            elif min_ped <= DIST_NEAR_PURE:
                alpha = 1.0
            else:
                alpha = (DIST_FAR_PURE - min_ped) / (DIST_FAR_PURE - DIST_NEAR_PURE)
        else:
            # apf / residual：按行人距离动态调整
            if min_ped >= DIST_FAR:
                alpha = 0.0
            elif min_ped <= DIST_NEAR:
                alpha = ALPHA_MAX
            else:
                alpha = ALPHA_MAX * (DIST_FAR - min_ped) / (DIST_FAR - DIST_NEAR)

        # 墙安全：很近时 APF 接管（保留兜底）
        if front_min < WALL_HARD_CUTOFF:
            alpha = 0.0
        elif FRONT_BLOCK_THRESH is not None and front_min < FRONT_BLOCK_THRESH:
            cap = WALL_SOFT_CAP if SAC_MODE == "residual" else 0.3
            alpha = min(alpha, cap)

        # 振荡兜底：pure_sac 模式下 SAC 输出 yaw 突变 > OSCILLATION_THRESH 时强制 α=0
        if SAC_MODE == "pure_sac" and self._sac_oscillating:
            alpha = 0.0

        # ----- 三种模式下的 (final_speed, final_heading) -----
        if SAC_MODE == "apf":
            # 纯 APF
            f_spd = float(np.clip(spd_apf, 0.10*self.pref_speed, self.pref_speed))
            f_hd = wrap_train(hd_apf)

        elif SAC_MODE == "residual":
            # APF + SAC 残差（原方案）
            ds_raw = a0 * 2.0 * self.pref_speed
            dh_raw = a1 * 2.0
            dh_clip = float(np.clip(dh_raw, -DH_CLIP, DH_CLIP))
            ds_clip = float(np.clip(ds_raw, DS_CLIP_MIN*self.pref_speed, DS_CLIP_MAX*self.pref_speed))
            f_spd = float(np.clip(spd_apf + alpha*ds_clip, 0.10*self.pref_speed, self.pref_speed))
            f_hd = wrap_train(hd_apf + alpha*dh_clip)

        else:  # pure_sac
            # 训练 action space: speed ∈ [0, 2.0]（m/s 绝对量），heading_delta ∈ [-π, π]
            # UnicycleDynamics: selected_heading = wrap(action[1] + heading_global_frame)
            # 所以 action[1] 是相对当前朝向的转角增量，不是绝对全局 heading！
            sac_speed = float(np.clip(a0, 0.0, self.pref_speed))
            sac_heading = wrap(self.cyaw + a1)   # ★ 关键修正：加上当前朝向
            # APF fallback：远距离时仍要跟 A*（训练里没见过内部障碍 + 长距离导航）
            apf_speed = float(np.clip(spd_apf, 0.10*self.pref_speed, self.pref_speed))
            apf_heading = wrap_train(hd_apf)
            # α 插值：α=1 完全 SAC，α=0 完全 APF
            f_spd = (1.0 - alpha) * apf_speed + alpha * sac_speed
            # ★ 当 α 较大（SAC 主导）时不强制最低速；α 小（APF 主导）时保持最低 0.1 防卡死
            if alpha > 0.5:
                f_spd = float(np.clip(f_spd, 0.0, self.pref_speed))
            else:
                f_spd = float(np.clip(f_spd, 0.10*self.pref_speed, self.pref_speed))
            # 朝向用单位向量插值（避免环绕）
            vx = (1.0 - alpha) * math.cos(apf_heading) + alpha * math.cos(sac_heading)
            vy = (1.0 - alpha) * math.sin(apf_heading) + alpha * math.sin(sac_heading)
            if vx*vx + vy*vy < 1e-8:
                f_hd = apf_heading
            else:
                f_hd = math.atan2(vy, vx)

        # 转向减速（三模式共用）
        herr = wrap(f_hd - self.cyaw)
        ae = abs(herr)
        v = f_spd
        if alpha > 0.1:
            if ae > 1.2: v *= 0.4
            elif ae > 0.6: v *= 0.6
            # pure_sac 主导（α>0.5）时不强制最低速，让 SAC 能"停下"
            if not (SAC_MODE == "pure_sac" and alpha > 0.5):
                v = max(v, 0.08 * self.pref_speed)
        else:
            if ae > 0.8: v *= 0.2
            elif ae > 0.4: v *= 0.4

        if dwp < 1.5 and not (SAC_MODE == "pure_sac" and alpha > 0.5):
            v = max(v, 0.10 * self.pref_speed)

        if front_min < 0.3:
            v = min(v, 0.03 * self.pref_speed / 0.26)
        elif front_min < 0.5:
            v = min(v, 0.10 * self.pref_speed / 0.26)

        final_speed = float(np.clip(v, 0.0, self.pref_speed))
        final_heading = wrap(f_hd)

        # 预测式防撞兜底（保留）
        def _would_collide(spd, hd):
            nx = self.cx + spd * math.cos(hd) * DT
            ny = self.cy + spd * math.sin(hd) * DT
            return is_robot_collided_with_wall(self.collision_grid, nx, ny, self.radius)

        if _would_collide(final_speed, final_heading):
            safe_spd = min(spd_apf, self.pref_speed)
            if not _would_collide(safe_spd, hd_apf):
                final_speed, final_heading = safe_spd, hd_apf
            else:
                found = False
                for k in (0.5, 0.25, 0.1):
                    s_try = safe_spd * k
                    if not _would_collide(s_try, hd_apf):
                        final_speed, final_heading = s_try, hd_apf
                        found = True
                        break
                if not found:
                    final_speed, final_heading = 0.0, self.cyaw

        # 推进
        self.cx += final_speed * math.cos(final_heading) * DT
        self.cy += final_speed * math.sin(final_heading) * DT
        self.cyaw = final_heading
        self.cv = final_speed

        if is_robot_collided_with_wall(self.collision_grid, self.cx, self.cy, self.radius):
            return 'collision_wall', final_speed, 0.0

        return 'running', final_speed, 0.0


# ============================================================
# 单回合
# ============================================================

def run_episode(controller, ep_idx, verbose=False, trace=False):
    peds = [Pedestrian(spec, speed_jitter=PED_SPEED_JITTER,
                       speed_multiplier=PED_SPEED_MULTIPLIER)
            for spec in PEDESTRIAN_SPECS]

    controller.reset()
    trace_buf = [] if trace else None
    graze_steps = 0
    min_ped_ever = float('inf')

    for step in range(MAX_STEPS):
        for p in peds:
            p.step(DT)

        for p in peds:
            d = math.hypot(p.pos[0]-controller.cx, p.pos[1]-controller.cy)
            if d < min_ped_ever:
                min_ped_ever = d
            if d < controller.radius + p.radius:
                graze_steps += 1
                break

        status, v, w = controller.step(peds)

        if status in ('reached', 'collision_ped', 'collision_wall'):
            if status == 'reached':
                return 'success', step+1, (step+1)*DT, controller.replan_count, -1, graze_steps, min_ped_ever
            elif status == 'collision_ped':
                min_idx, min_d = -1, float('inf')
                for i, p in enumerate(peds):
                    d = math.hypot(p.pos[0]-controller.cx, p.pos[1]-controller.cy)
                    if d < min_d:
                        min_d, min_idx = d, i
                return 'collision_ped', step+1, (step+1)*DT, controller.replan_count, min_idx, graze_steps, min_ped_ever
            else:
                return 'collision_wall', step+1, (step+1)*DT, controller.replan_count, -1, graze_steps, min_ped_ever

    return 'timeout', MAX_STEPS, MAX_STEPS*DT, controller.replan_count, -1, graze_steps, min_ped_ever


# ============================================================
# 主入口
# ============================================================

def main():
    from stable_baselines3 import SAC

    print("=" * 70)
    print("📊 复杂地图大规模批量测试")
    mode_desc = {
        "apf":      "纯 APF 基线（不调用 SAC）",
        "residual": "APF + SAC 残差（原方案）",
        "pure_sac": "纯 SAC（接近障碍时完全 SAC，远距离仍用 APF 跟 A*）",
    }
    print(f"   模式: SAC_MODE={SAC_MODE}  →  {mode_desc[SAC_MODE]}")
    print(f"   起点: {START_XY}  终点: {GOAL_XY}")
    print(f"   回合数: {TEST_EPISODES}")
    print(f"   行人速度随机化: ±{int(PED_SPEED_JITTER*100)}%")
    print(f"   行人速度倍数: ×{PED_SPEED_MULTIPLIER:.1f}")
    print(f"   单回合超时: {MAX_STEPS*DT:.0f}s ({MAX_STEPS} 步)")
    if SAC_MODE == "pure_sac":
        print(f"   纯SAC转向范围: ±{math.degrees(PURE_SAC_HEADING_RANGE):.0f}°")
    print("=" * 70)

    print(">>> 初始化 A* ...")
    planner = GlobalPlanner(resolution=0.2, robot_radius=ROBOT_RADIUS)
    print(f"    地图: {planner.grid_map.width}×{planner.grid_map.height}")

    print(">>> 提取颗粒墙...")
    wall_pts = build_wall_points(planner.grid_map, subsample=1)
    print(f"    颗粒墙点数: {len(wall_pts)}")

    print(">>> 构建未膨胀碰撞地图...")
    collision_grid = build_collision_grid()

    if SAC_MODE == "pure_sac":
        model_path = MODEL_PATH_PURE_SAC
    else:
        model_path = MODEL_PATH
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ 模型文件不存在: {model_path}.zip")
        sys.exit(1)
    print(f">>> 加载 SAC 模型: {os.path.basename(model_path)}")
    model = SAC.load(model_path)

    controller = DeployController(model, planner, wall_pts, collision_grid)
    print(f">>> 初始 A* waypoints: {len(controller.waypoints)} 个")

    print(f"\n>>> 开始 {TEST_EPISODES} 回合测试...")
    print("-" * 70)

    results = {'success': 0, 'collision_ped': 0, 'collision_wall': 0, 'timeout': 0}
    times_success = []
    steps_all = []
    replans_all = []
    ped_collide_count = [0] * 5
    graze_all = []
    minped_all = []

    t0 = time.time()
    for ep in range(TEST_EPISODES):
        status, steps, t_sec, replans, ped_idx, graze, minped = run_episode(
            controller, ep, trace=False)
        results[status] += 1
        steps_all.append(steps)
        replans_all.append(replans)
        graze_all.append(graze)
        minped_all.append(minped)
        if status == 'success':
            times_success.append(t_sec)
        if status == 'collision_ped' and 0 <= ped_idx < 5:
            ped_collide_count[ped_idx] += 1

        icon = {'success': '✅', 'collision_ped': '💥',
                'collision_wall': '🧱', 'timeout': '⏱️'}[status]
        extra = f" hit=P{ped_idx+1}" if status == 'collision_ped' else ""
        print(f"  Ep {ep+1:03d} {icon} {status:14s} steps={steps:4d} t={t_sec:5.1f}s "
              f"replan={replans} final=({controller.cx:+.1f},{controller.cy:+.1f}) "
              f"minped={minped:.2f}m graze={graze}{extra}")

        if (ep + 1) % 50 == 0:
            sr = results['success'] / (ep+1) * 100
            elapsed = time.time() - t0
            eta = elapsed / (ep+1) * (TEST_EPISODES - ep - 1)
            print(f"  -- [中间汇总] {ep+1} 回合: 成功率 {sr:.1f}% | "
                  f"已用 {elapsed:.0f}s | 预计剩余 {eta:.0f}s --")

    total_time = time.time() - t0

    print("\n" + "=" * 70)
    print(f"🏆 最终测试报告  (SAC_MODE={SAC_MODE})")
    print("=" * 70)
    n = TEST_EPISODES
    print(f"  ✅ 成功:       {results['success']:3d} / {n}  ({results['success']/n*100:5.1f}%)")
    print(f"  💥 行人碰撞:   {results['collision_ped']:3d} / {n}  ({results['collision_ped']/n*100:5.1f}%)")
    print(f"  🧱 撞墙:       {results['collision_wall']:3d} / {n}  ({results['collision_wall']/n*100:5.1f}%)")
    print(f"  ⏱️ 超时:       {results['timeout']:3d} / {n}  ({results['timeout']/n*100:5.1f}%)")
    print("-" * 70)
    if times_success:
        print(f"  📏 成功回合平均到达时间: {np.mean(times_success):.2f}s "
              f"(std {np.std(times_success):.2f}, "
              f"min {min(times_success):.1f}, max {max(times_success):.1f})")
    print(f"  👣 全部回合平均步数:      {np.mean(steps_all):.1f}")
    print(f"  🔄 平均重规划次数:        {np.mean(replans_all):.2f}")
    print(f"  🤏 平均擦碰步数:          {np.mean(graze_all):.2f}")
    print(f"  📐 回合内行人最近距离均值: {np.mean(minped_all):.2f}m")
    print(f"  ⏰ 总测试耗时:            {total_time:.1f}s ({total_time/n:.2f}s/回合)")

    p = results['success'] / n
    se = math.sqrt(p * (1-p) / n)
    ci_low = max(0, p - 1.96*se) * 100
    ci_high = min(1, p + 1.96*se) * 100
    print(f"  📊 成功率 95% CI:         [{ci_low:.1f}%, {ci_high:.1f}%]")

    if results['collision_ped'] > 0:
        print(f"\n  💥 行人碰撞分布（共 {results['collision_ped']} 次）:")
        for i in range(5):
            if ped_collide_count[i] > 0:
                pct = ped_collide_count[i] / results['collision_ped'] * 100
                print(f"    P{i+1}: {ped_collide_count[i]:3d} 次 ({pct:5.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()