#!/usr/bin/env python3
"""
行人移动控制器 v3
发布完整行人状态到 /pedestrian_states，供部署节点构建观测。

Float32MultiArray 格式 (每个行人9个值，共27个):
  [x, y, vx, vy, speed, goal_x, goal_y, pref_speed, radius] × 3

用法: python3 pedestrian_mover.py
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import subprocess
import math


class PedestrianMover(Node):
    def __init__(self):
        super().__init__('pedestrian_mover')

        self.pub_states = self.create_publisher(Float32MultiArray, '/pedestrian_states', 10)

        # 行人配置
        self.pedestrians = [
            {
                'name': 'pedestrian_1',
                'waypoints': [(1.0, 0.0), (1.0, 4.0), (-2.0, 4.0), (-2.0, 0.0)],
                'pref_speed': 0.8,
                'radius': 0.3,
                'x': 1.0, 'y': 0.0,
                'vx': 0.0, 'vy': 0.0,
                'wp_idx': 0,
            },
            {
                'name': 'pedestrian_2',
                'waypoints': [(-1.0, 2.0), (3.0, 2.0), (3.0, -3.0), (-1.0, -3.0)],
                'pref_speed': 0.6,
                'radius': 0.3,
                'x': -1.0, 'y': 2.0,
                'vx': 0.0, 'vy': 0.0,
                'wp_idx': 0,
            },
            {
                'name': 'pedestrian_3',
                'waypoints': [(2.0, -2.0), (-3.0, -2.0), (-3.0, 3.0), (2.0, 3.0)],
                'pref_speed': 0.7,
                'radius': 0.3,
                'x': 2.0, 'y': -2.0,
                'vx': 0.0, 'vy': 0.0,
                'wp_idx': 0,
            },
        ]

        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.update_pedestrians)
        self.get_logger().info('🚶 行人控制器 v3 启动！发布到 /pedestrian_states (含目标点)')

    def update_pedestrians(self):
        for ped in self.pedestrians:
            wp = ped['waypoints'][ped['wp_idx']]
            tx, ty = wp

            dx = tx - ped['x']
            dy = ty - ped['y']
            dist = math.hypot(dx, dy)

            if dist < 0.2:
                ped['wp_idx'] = (ped['wp_idx'] + 1) % len(ped['waypoints'])
                ped['vx'] = 0.0
                ped['vy'] = 0.0
                continue

            ped['vx'] = (dx / dist) * ped['pref_speed']
            ped['vy'] = (dy / dist) * ped['pref_speed']
            ped['x'] += ped['vx'] * self.dt
            ped['y'] += ped['vy'] * self.dt

            yaw = math.atan2(dy, dx)
            self.set_pose(ped['name'], ped['x'], ped['y'], 0.5, yaw)

        # === 发布: [x,y,vx,vy,speed,goal_x,goal_y,pref_speed,radius] × 3 ===
        msg = Float32MultiArray()
        data = []
        for ped in self.pedestrians:
            # 当前目标路点作为 goal
            wp = ped['waypoints'][ped['wp_idx']]
            data.extend([
                ped['x'], ped['y'],             # 位置
                ped['vx'], ped['vy'],            # 速度
                ped['pref_speed'],               # 期望速度
                wp[0], wp[1],                    # 当前目标路点
                ped['pref_speed'],               # pref_speed (训练时行人的 pref_speed)
                ped['radius'],                   # radius
            ])
        msg.data = data
        self.pub_states.publish(msg)

    def set_pose(self, model_name, x, y, z, yaw):
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        cmd = (
            f'gz service -s /world/training_world_12x12/set_pose '
            f'--reqtype gz.msgs.Pose '
            f'--reptype gz.msgs.Boolean '
            f'--timeout 100 '
            f'--req "name: \'{model_name}\', '
            f'position: {{x: {x}, y: {y}, z: {z}}}, '
            f'orientation: {{w: {qw}, x: 0.0, y: 0.0, z: {qz}}}"'
        )
        try:
            subprocess.run(cmd, shell=True, capture_output=True, timeout=0.5)
        except:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = PedestrianMover()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()