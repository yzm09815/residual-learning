#!/usr/bin/env python3
"""
纯 APF (人工势场法) 导航节点
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import numpy as np
import math

def normalize_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

class APFDeployNode(Node):
    def __init__(self):
        super().__init__('apf_deploy_node')


        # === 订阅与发布 ===
        #发布者 (Publisher)：发送控制指令
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        #订阅者 (Subscription)：获取激光雷达数据
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        #订阅者 (Subscription)：获取里程计数据
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # === 目标与 APF 参数 ===
        self.target_x = 4.0
        self.target_y = 4.0
        self.pref_speed = 1.0
        self.radius = 0.3
        self.k_att = 1.0
        self.k_rep = 0.3      # 降低斥力系数
        self.d0 = 1.5         # 缩小斥力影响范围
        self.max_rep = 5.0    # 单次斥力上限

        # === 运动学限制 ===
        self.max_v = 0.22
        self.max_w = 1.82

        # === 状态变量 ===
        self.cx = 0.0
        self.cy = 0.0
        self.cyaw = 0.0
        self.prev_w = 0.0

        self.get_logger().info(f'🎯 纯 APF 导航已启动 | 目标: ({self.target_x},{self.target_y})')

    def cmd(self, v, w):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_footprint'
        m.twist.linear.x = float(v)
        m.twist.angular.z = float(w)
        self.pub.publish(m)

    def odom_cb(self, msg):
        self.cx = msg.pose.pose.position.x
        self.cy = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.cyaw = math.atan2(2.0*(q.w*q.z+q.x*q.y), 1.0-2.0*(q.y*q.y+q.z*q.z))

    def scan_cb(self, msg):
        dx = self.target_x - self.cx
        dy = self.target_y - self.cy
        dist = math.hypot(dx, dy)

        if dist < 0.3:
            self.get_logger().info('🎉 到达目标点！')
            self.cmd(0.0, 0.0)
            return

        # === APF ===
        pos = np.array([self.cx, self.cy])
        goal = np.array([self.target_x, self.target_y])
        gv = goal - pos

        # 1. 引力
        f_att = self.k_att * (gv / dist) * self.pref_speed

        # 2. 斥力
        f_rep = np.zeros(2)
        ranges = np.array(msg.ranges)
        front_min = 5.0
        for i, r in enumerate(ranges):
            if math.isnan(r) or math.isinf(r) or r <= 0:
                continue
            sr = max(float(r), 0.15)
            al = msg.angle_min + i * msg.angle_increment
            if al > math.pi:
                al -= 2 * math.pi

            if abs(al) < 0.4:
                front_min = min(front_min, sr)

            if sr < self.d0:
                ag = al + self.cyaw
                op = pos + sr * np.array([math.cos(ag), math.sin(ag)])
                diff = pos - op
                d = np.linalg.norm(diff)
                de = max(d - self.radius - 0.3, 0.05)
                if de < self.d0 and d > 0.001:
                    mag = self.k_rep * (1.0/de - 1.0/self.d0) / (de**2)
                    mag = min(mag, self.max_rep)  # 截断斥力上限
                    f_rep += mag * (diff / d)

        # 3. 合力
        va = f_att + f_rep
        spd_apf = np.linalg.norm(va)
        hd_apf = math.atan2(va[1], va[0])

        f_spd = float(np.clip(spd_apf, 0, self.pref_speed))
        f_hd = hd_apf

        # === 底盘 v,w 转换与平滑 ===
        herr = normalize_angle(f_hd - self.cyaw)

        # 角速度：增大增益，减弱滤波延迟
        w = 3.0 * herr
        w = 0.7 * w + 0.3 * self.prev_w
        self.prev_w = w

        # 线速度
        v = f_spd * (self.max_v / self.pref_speed)
        ae = abs(herr)
        if ae > 0.8:
            v *= 0.1    # 大角度误差时几乎停下来原地转
        elif ae > 0.4:
            v *= 0.3

        if dist < 1.5:
            v = max(v, 0.15)

        # 紧急避障
        if front_min < 0.3:
            v = min(v, 0.03)
        elif front_min < 0.5:
            v = min(v, 0.10)

        # === 最终限幅与发布 ===
        v = float(np.clip(v, 0.0, self.max_v))
        w = float(np.clip(w, -self.max_w, self.max_w))
        self.cmd(v, w)

        self.get_logger().info(
            f'({self.cx:.1f},{self.cy:.1f}) 距:{dist:.1f}m | '
            f'APF:h={math.degrees(hd_apf):.0f}° | '
            f'v={v:.2f},w={w:.2f} | 前:{front_min:.1f}m'
        )

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(APFDeployNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()