#!/usr/bin/env python3
"""
全局规划模块 — A* + 栅格地图 + Waypoint 提取

功能：
  1. 定义复杂地图的静态障碍物（与 Gazebo world 文件一致）
  2. A* 在栅格地图上搜索最短路径
  3. 路径简化：去掉共线点，只保留拐角作为 waypoints
  4. 动态重规划：行人堵路时临时加入行人位置，重新搜索

地图坐标系：与 Gazebo 世界坐标一致
  原点在地图中心，X 向右，Y 向上
  地图范围: [-10, 10] × [-10, 10]（20m × 20m）
"""

import numpy as np
import heapq
import math


# ============================================================
#  1. 栅格地图（与 Gazebo world 一致）
# ============================================================

class GridMap:
    """
    栅格地图：将连续世界坐标离散化为格子
    resolution: 每个格子的边长（米），越小越精细但计算越慢
    """

    def __init__(self, x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0,
                 resolution=0.2, robot_radius=0.3):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.resolution = resolution
        self.robot_radius = robot_radius

        # 膨胀半径：机器人半径 + 安全余量
        self.inflate_radius = robot_radius + 0.10

        self.width = int((x_max - x_min) / resolution)
        self.height = int((y_max - y_min) / resolution)

        # 0 = free, 1 = obstacle
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

        # 加载静态障碍物
        self._load_walls()

    def _load_walls(self):
        """
        定义所有静态墙壁 — 与 Gazebo world 文件完全一致
        每堵墙用 (cx, cy, length, thickness, angle_deg) 描述
          cx, cy: 墙壁中心世界坐标
          length: 墙壁长度
          thickness: 墙壁厚度
          angle_deg: 旋转角度（0=沿X轴，90=沿Y轴）
        """
        walls = [
            # ===== 外围边界墙 =====
            (0.0, 10.0, 20.4, 0.2, 0),    # 上边界
            (0.0, -10.0, 20.4, 0.2, 0),   # 下边界
            (-10.0, 0.0, 20.4, 0.2, 90),  # 左边界
            (10.0, 0.0, 20.4, 0.2, 90),   # 右边界

            # ===== 中央横墙 y=2（中间留 3m 缺口 x∈[1,4]）=====
            (-3.5, 2.0, 7.0, 0.2, 0),     # 左半段 x: [-7, 0]
            (6.5, 2.0, 5.0, 0.2, 0),      # 右半段 x: [4, 9]

            # ===== 下方横墙 y=-3（中间留 3m 缺口 x∈[-1.5,1.5]）=====
            (-5.5, -3.0, 5.0, 0.2, 0),    # 左段 x: [-8, -3]
            (5.5, -3.0, 5.0, 0.2, 0),     # 右段 x: [3, 8]

            # ===== U形陷阱（左下角，开口朝上）=====
            (-7.5, -6.5, 3.0, 0.2, 90),   # U形左壁 x=-7.5, y: [-8, -5]
            (-6.0, -8.0, 3.0, 0.2, 0),    # U形底壁 y=-8, x: [-7.5, -4.5]
            (-4.5, -6.5, 3.0, 0.2, 90),   # U形右壁 x=-4.5, y: [-8, -5]

            # ===== 竖墙 =====
            (0.0, -6.5, 4.0, 0.2, 90),    # 中央竖墙 x=0
            (7.0, 5.5, 5.0, 0.2, 90),     # 右侧竖墙 x=7

            # ===== 方柱 =====
            (3.5, 0.0, 0.8, 0.8, 0),      # 方柱1
            (-3.0, -1.0, 0.8, 0.8, 0),    # 方柱2
        ]

        for wall in walls:
            self._add_wall(*wall)

    def _add_wall(self, cx, cy, length, thickness, angle_deg):
        """将一堵墙栅格化并膨胀后写入 grid"""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # 膨胀后的半长和半宽
        half_l = length / 2.0 + self.inflate_radius
        half_t = thickness / 2.0 + self.inflate_radius

        # 扫描墙壁覆盖的栅格范围
        # 先算旋转后的包围盒
        corners = [
            (cx + cos_a * dx - sin_a * dy, cy + sin_a * dx + cos_a * dy)
            for dx in [-half_l, half_l]
            for dy in [-half_t, half_t]
        ]
        min_x = min(c[0] for c in corners) - 0.1
        max_x = max(c[0] for c in corners) + 0.1
        min_y = min(c[1] for c in corners) - 0.1
        max_y = max(c[1] for c in corners) + 0.1

        # 遍历包围盒内的每个格子
        for gy in range(max(0, self.world_to_grid_y(min_y)),
                        min(self.height, self.world_to_grid_y(max_y) + 1)):
            for gx in range(max(0, self.world_to_grid_x(min_x)),
                            min(self.width, self.world_to_grid_x(max_x) + 1)):
                # 格子中心的世界坐标
                wx = self.grid_to_world_x(gx)
                wy = self.grid_to_world_y(gy)

                # 转换到墙壁的局部坐标系
                dx = wx - cx
                dy = wy - cy
                local_x = cos_a * dx + sin_a * dy
                local_y = -sin_a * dx + cos_a * dy

                # 判断是否在膨胀后的墙壁范围内
                if abs(local_x) <= half_l and abs(local_y) <= half_t:
                    self.grid[gy, gx] = 1

    # ---- 坐标转换 ----
    def world_to_grid_x(self, wx):
        return int((wx - self.x_min) / self.resolution)

    def world_to_grid_y(self, wy):
        return int((wy - self.y_min) / self.resolution)

    def world_to_grid(self, wx, wy):
        return self.world_to_grid_x(wx), self.world_to_grid_y(wy)

    def grid_to_world_x(self, gx):
        return self.x_min + (gx + 0.5) * self.resolution

    def grid_to_world_y(self, gy):
        return self.y_min + (gy + 0.5) * self.resolution

    def grid_to_world(self, gx, gy):
        return self.grid_to_world_x(gx), self.grid_to_world_y(gy)

    def is_free(self, gx, gy):
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return self.grid[gy, gx] == 0
        return False

    def add_temp_obstacle(self, wx, wy, radius=0.8):
        """临时加入动态障碍物（行人堵路时用），返回受影响的格子列表以便之后清除"""
        affected = []
        inflate_r = radius + self.inflate_radius
        gx_c, gy_c = self.world_to_grid(wx, wy)
        r_cells = int(inflate_r / self.resolution) + 1

        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                gx = gx_c + dx
                gy = gy_c + dy
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    cell_wx = self.grid_to_world_x(gx)
                    cell_wy = self.grid_to_world_y(gy)
                    dist = math.hypot(cell_wx - wx, cell_wy - wy)
                    if dist <= inflate_r:
                        if self.grid[gy, gx] == 0:  # 只记录原本是free的
                            affected.append((gx, gy))
                        self.grid[gy, gx] = 1
        return affected

    def remove_temp_obstacle(self, affected_cells):
        """清除临时障碍物"""
        for gx, gy in affected_cells:
            self.grid[gy, gx] = 0


# ============================================================
#  2. A* 搜索
# ============================================================

def astar(grid_map, start_world, goal_world):
    """
    A* 搜索

    参数:
        grid_map: GridMap 实例
        start_world: (x, y) 世界坐标
        goal_world: (x, y) 世界坐标

    返回:
        path: [(x1,y1), (x2,y2), ...] 世界坐标路径，或 None（无解）
    """
    sx, sy = grid_map.world_to_grid(*start_world)
    gx, gy = grid_map.world_to_grid(*goal_world)

    # 检查起点终点是否合法
    if not grid_map.is_free(sx, sy):
        # 起点在障碍物内，尝试找最近的free格子
        sx, sy = _find_nearest_free(grid_map, sx, sy)
        if sx is None:
            return None

    if not grid_map.is_free(gx, gy):
        gx, gy = _find_nearest_free(grid_map, gx, gy)
        if gx is None:
            return None

    # 8方向移动
    neighbors = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414),
    ]

    # open_set: (f_score, counter, gx, gy)
    counter = 0
    open_set = [(0, counter, sx, sy)]
    came_from = {}
    g_score = {(sx, sy): 0.0}
    closed_set = set()

    while open_set:
        f, _, cx, cy = heapq.heappop(open_set)

        if (cx, cy) in closed_set:
            continue
        closed_set.add((cx, cy))

        # 到达目标
        if cx == gx and cy == gy:
            # 回溯路径
            path_grid = []
            node = (gx, gy)
            while node in came_from:
                path_grid.append(node)
                node = came_from[node]
            path_grid.append((sx, sy))
            path_grid.reverse()

            # 转换为世界坐标
            path_world = [grid_map.grid_to_world(px, py) for px, py in path_grid]
            return path_world

        for dx, dy, cost in neighbors:
            nx, ny = cx + dx, cy + dy
            if not grid_map.is_free(nx, ny):
                continue
            if (nx, ny) in closed_set:
                continue

            # 对角线移动时检查是否会切角穿墙
            if dx != 0 and dy != 0:
                if not grid_map.is_free(cx + dx, cy) or not grid_map.is_free(cx, cy + dy):
                    continue

            new_g = g_score[(cx, cy)] + cost
            if new_g < g_score.get((nx, ny), float('inf')):
                g_score[(nx, ny)] = new_g
                h = math.hypot(nx - gx, ny - gy)  # 欧几里得启发式
                f = new_g + h
                came_from[(nx, ny)] = (cx, cy)
                counter += 1
                heapq.heappush(open_set, (f, counter, nx, ny))

    return None  # 无解


def _find_nearest_free(grid_map, gx, gy, max_search=20):
    """在起点/终点被障碍物覆盖时，找最近的free格子"""
    for r in range(1, max_search):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) == r or abs(dy) == r:
                    nx, ny = gx + dx, gy + dy
                    if grid_map.is_free(nx, ny):
                        return nx, ny
    return None, None


# ============================================================
#  3. 路径简化：只保留拐点作为 Waypoints
# ============================================================

def simplify_path(path, min_spacing=1.0):
    """
    简化路径：
    1. 去掉共线的中间点（只保留方向改变的拐角）
    2. 确保相邻 waypoint 间距不超过 max_spacing
    3. 确保相邻 waypoint 间距不小于 min_spacing（合并太近的点）

    参数:
        path: [(x, y), ...] A* 输出的完整路径
        min_spacing: waypoint 最小间距（米）

    返回:
        waypoints: [(x, y), ...] 简化后的路径
    """
    if path is None or len(path) <= 2:
        return path

    # 第一步：去掉共线点
    key_points = [path[0]]
    for i in range(1, len(path) - 1):
        # 检查 path[i] 是否是拐点
        dx1 = path[i][0] - path[i - 1][0]
        dy1 = path[i][1] - path[i - 1][1]
        dx2 = path[i + 1][0] - path[i][0]
        dy2 = path[i + 1][1] - path[i][1]

        # 方向改变了就保留
        if abs(dx1 * dy2 - dy1 * dx2) > 1e-6:
            key_points.append(path[i])
    key_points.append(path[-1])

    # 第二步：合并太近的点
    if len(key_points) <= 2:
        return key_points

    merged = [key_points[0]]
    for i in range(1, len(key_points) - 1):
        dist = math.hypot(key_points[i][0] - merged[-1][0],
                          key_points[i][1] - merged[-1][1])
        if dist >= min_spacing:
            merged.append(key_points[i])
    merged.append(key_points[-1])

    return merged


# ============================================================
#  4. 主接口：一键规划
# ============================================================

class GlobalPlanner:
    """
    全局规划器：封装地图 + A* + waypoint

    使用方法:
        planner = GlobalPlanner()
        waypoints = planner.plan(start=(sx, sy), goal=(gx, gy))
        # waypoints = [(x1,y1), (x2,y2), ..., (gx,gy)]
    """

    def __init__(self, resolution=0.2, robot_radius=0.3):
        self.grid_map = GridMap(resolution=resolution, robot_radius=robot_radius)
        self.current_waypoints = []
        self.current_wp_index = 0
        self._temp_obstacles = []  # 临时障碍物记录

    def plan(self, start, goal, min_spacing=1.0):
        """
        规划全局路径

        参数:
            start: (x, y) 起点世界坐标
            goal: (x, y) 终点世界坐标
            min_spacing: waypoint 最小间距

        返回:
            waypoints: [(x, y), ...] 或 None
        """
        path = astar(self.grid_map, start, goal)
        if path is None:
            return None

        self.current_waypoints = simplify_path(path, min_spacing=min_spacing)
        self.current_wp_index = 0
        return self.current_waypoints

    def get_current_waypoint(self):
        """获取当前子目标"""
        if self.current_wp_index < len(self.current_waypoints):
            return self.current_waypoints[self.current_wp_index]
        return None

    def advance_waypoint(self, robot_pos, threshold=0.8):
        """
        检查是否到达当前 waypoint，是则切换到下一个

        参数:
            robot_pos: (x, y) 机器人当前位置
            threshold: 到达判定距离（米）

        返回:
            switched: bool 是否切换了
        """
        wp = self.get_current_waypoint()
        if wp is None:
            return False

        dist = math.hypot(robot_pos[0] - wp[0], robot_pos[1] - wp[1])
        if dist < threshold:
            self.current_wp_index += 1
            return True
        return False

    def is_goal_reached(self):
        """是否已经到达/通过所有 waypoints"""
        return self.current_wp_index >= len(self.current_waypoints)

    def replan_with_pedestrians(self, start, goal, ped_positions, ped_radius=0.8):
        """
        动态重规划：把行人当前位置临时加入障碍物，重新搜索

        参数:
            start: 机器人当前位置
            goal: 最终目标
            ped_positions: [(x, y), ...] 行人位置列表
            ped_radius: 行人膨胀半径

        返回:
            waypoints 或 None
        """
        # 清除上次的临时障碍物
        self._clear_temp_obstacles()

        # 加入行人作为临时障碍物
        for px, py in ped_positions:
            affected = self.grid_map.add_temp_obstacle(px, py, radius=ped_radius)
            self._temp_obstacles.append(affected)

        # 重新规划
        result = self.plan(start, goal)

        # 清除临时障碍物（保持地图干净）
        self._clear_temp_obstacles()

        return result

    def _clear_temp_obstacles(self):
        for affected in self._temp_obstacles:
            self.grid_map.remove_temp_obstacle(affected)
        self._temp_obstacles = []

    def get_remaining_waypoints(self):
        """获取剩余的所有 waypoints（用于可视化）"""
        return self.current_waypoints[self.current_wp_index:]


# ============================================================
#  测试入口
# ============================================================
if __name__ == '__main__':
    planner = GlobalPlanner()

    # 测试：从左下到右上
    start = (-8.0, -8.0)
    goal = (8.0, 8.0)

    print(f"起点: {start}")
    print(f"终点: {goal}")
    print(f"地图大小: {planner.grid_map.width}×{planner.grid_map.height}")
    print(f"障碍物占比: {planner.grid_map.grid.sum() / planner.grid_map.grid.size * 100:.1f}%")

    waypoints = planner.plan(start, goal)
    if waypoints:
        print(f"\n✅ 规划成功！{len(waypoints)} 个 waypoints:")
        for i, wp in enumerate(waypoints):
            print(f"  wp{i}: ({wp[0]:.1f}, {wp[1]:.1f})")
    else:
        print("❌ 规划失败，无可行路径")

    # 测试动态重规划
    print("\n--- 测试动态重规划（行人堵路）---")
    ped_positions = [(2.0, 2.5), (-1.0, -2.5)]
    waypoints2 = planner.replan_with_pedestrians(start, goal, ped_positions)
    if waypoints2:
        print(f"✅ 重规划成功！{len(waypoints2)} 个 waypoints")
    else:
        print("❌ 重规划失败")
