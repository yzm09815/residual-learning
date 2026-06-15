#!/usr/bin/env python3
"""
复杂地图行人控制节点 — v3 三段式遭遇版

路径 (-8,-8) → (8,8)，沿途三个遭遇区：

  Zone1（起点附近）:
    P1: 下方缺口(y=-3)左半段巡逻，x:-2.5~0.5
    机器人穿缺口时遭遇一次

  Zone2（中部，3个行人密集）:
    P2: 两缺口之间对角穿越，(-2,-2)→(3,1)
    P3: 中央缺口(y=2)右半段巡逻，x:2~3.8
    P4: 纵向穿越 x≈2, y:-2~1.5
    机器人在中段路径遭遇3个行人（RL优势最大）

  Zone3（终点附近）:
    P5: 右上区域 x:5~8, y:6~8 巡逻
    机器人快到终点时遭遇一次

流程: APF → P1(RL) → APF → P2+P3+P4(RL) → APF → APF → P5(RL) → 终点
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import subprocess
import numpy as np


class PedestrianMoverComplex(Node):
    def __init__(self):
        super().__init__('pedestrian_mover_complex')

        self.pub = self.create_publisher(Float32MultiArray, '/pedestrian_states', 10)

        self.dt = 0.05  # 20 Hz
        self.timer = self.create_timer(self.dt, self.timer_cb)

        self.pedestrians = [
            # ===== Zone 1: 起点附近 =====
            {
                'name': 'pedestrian_1',
                'radius': 0.3,
                'pref_speed': 0.5,
                # P1: 下方缺口左半段巡逻 x:-2.5~0.5
                # 缺口右半段 x:0.5~3 空着让APF能通过
                'waypoints': [
                    (-2.5, -2.8),
                    (0.5, -2.8),
                    (0.5, -3.2),
                    (-2.5, -3.2),
                ],
                'current_wp': 0,
                'pos': np.array([-1.0, -3.0]),
                'vel': np.array([0.0, 0.0]),
            },

            # ===== Zone 2: 中部（3个行人密集区）=====
            {
                'name': 'pedestrian_2',
                'radius': 0.3,
                'pref_speed': 0.7,
                # P2: 两缺口之间对角穿越
                # 在 y:-2 到 y:1 之间活动，不进入缺口
                'waypoints': [
                    (-2.0, -2.0),
                    (3.0, 1.0),
                    (3.5, -1.0),
                    (-1.0, -2.5),
                ],
                'current_wp': 0,
                'pos': np.array([0.0, -0.5]),
                'vel': np.array([0.0, 0.0]),
            },
            {
                'name': 'pedestrian_3',
                'radius': 0.3,
                'pref_speed': 0.45,
                # P3: 中央缺口右半段巡逻 x:2~3.8
                # 缺口左半段 x:0~2 空着让APF能通过
                'waypoints': [
                    (2.0, 1.8),
                    (3.8, 1.8),
                    (3.8, 2.2),
                    (2.0, 2.2),
                ],
                'current_wp': 0,
                'pos': np.array([3.0, 2.0]),
                'vel': np.array([0.0, 0.0]),
            },
            {
                'name': 'pedestrian_4',
                'radius': 0.3,
                'pref_speed': 0.65,
                # P4: 纵向穿越 x≈2, 连接两缺口间的通道
                # 偏离缺口正中央，不直接堵口
                'waypoints': [
                    (2.0, -2.0),
                    (2.0, 1.5),
                    (2.5, 1.5),
                    (2.5, -2.0),
                ],
                'current_wp': 0,
                'pos': np.array([2.0, -0.5]),
                'vel': np.array([0.0, 0.0]),
            },

            # ===== Zone 3: 终点附近 =====
            {
                'name': 'pedestrian_5',
                'radius': 0.3,
                'pref_speed': 0.6,
                # P5: 右上区域巡逻 x:5~8, y:6~8
                # 机器人穿过中部后走向终点时遭遇
                'waypoints': [
                    (5.0, 7.0),
                    (8.0, 7.0),
                    (8.0, 6.0),
                    (5.0, 6.0),
                ],
                'current_wp': 0,
                'pos': np.array([6.5, 7.0]),
                'vel': np.array([0.0, 0.0]),
            },
        ]
        self.num_ped = len(self.pedestrians)
        self.world_name = 'complex_nav_world'
        self.step_count = 0

        self.get_logger().info(f'🚶 {self.num_ped}人行人控制器启动（v3 三段式遭遇版）')
        self.get_logger().info(f'  Zone1: P1(下方缺口) | Zone2: P2+P3+P4(中部) | Zone3: P5(终点)')
        for i, ped in enumerate(self.pedestrians):
            self.get_logger().info(
                f'  行人{i+1}: 速度={ped["pref_speed"]} '
                f'路径={[(f"{w[0]:.1f},{w[1]:.1f}") for w in ped["waypoints"]]}')

    def timer_cb(self):
        self.step_count += 1

        for ped in self.pedestrians:
            wp = ped['waypoints'][ped['current_wp']]
            target = np.array(wp)

            diff = target - ped['pos']
            dist = np.linalg.norm(diff)

            if dist < 0.3:
                ped['current_wp'] = (ped['current_wp'] + 1) % len(ped['waypoints'])
                wp = ped['waypoints'][ped['current_wp']]
                target = np.array(wp)
                diff = target - ped['pos']
                dist = np.linalg.norm(diff)

            if dist > 0.01:
                direction = diff / dist
                ped['vel'] = direction * ped['pref_speed']
            else:
                ped['vel'] = np.array([0.0, 0.0])

            ped['pos'] = ped['pos'] + ped['vel'] * self.dt
            self._set_gz_pose(ped['name'], ped['pos'][0], ped['pos'][1])

        self._publish_states()

        if self.step_count % 200 == 0:
            for i, ped in enumerate(self.pedestrians):
                self.get_logger().info(
                    f'  🚶 行人{i+1}: ({ped["pos"][0]:.1f}, {ped["pos"][1]:.1f}) '
                    f'wp={ped["current_wp"]}/{len(ped["waypoints"])}')

    def _set_gz_pose(self, model_name, x, y, z=0.5):
        cmd = [
            'gz', 'service',
            '-s', f'/world/{self.world_name}/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '100',
            '--req',
            f'name: "{model_name}", '
            f'position: {{x: {x}, y: {y}, z: {z}}}'
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=0.5)
        except (subprocess.TimeoutExpired, Exception):
            pass

    def _publish_states(self):
        msg = Float32MultiArray()
        data = []
        for ped in self.pedestrians:
            wp = ped['waypoints'][ped['current_wp']]
            speed = np.linalg.norm(ped['vel'])
            data.extend([
                ped['pos'][0], ped['pos'][1],
                ped['vel'][0], ped['vel'][1],
                speed,
                wp[0], wp[1],
                ped['pref_speed'],
                ped['radius'],
            ])
        msg.data = [float(d) for d in data]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PedestrianMoverComplex()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()