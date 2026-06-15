"""
Launch 文件: 12x12 围墙训练场景
用法: ros2 launch <your_package> launch_12x12.py
或者直接: python3 launch_12x12.py (如果用 ros2 launch 有问题)

注意: 
1. 请确认 training_world_12x12.sdf 的路径
2. 请确认你的 TurtleBot3 model 路径
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # ============================================
    # 路径配置 - 根据你的环境修改
    # ============================================
    
    # SDF 世界文件路径（把文件放到这个位置，或者修改路径）
    world_file = os.path.expanduser('~/ros2_ws/src/obstacle_avoidance/worlds/training_world_12x12.sdf')
    
    # TurtleBot3 相关
    TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'waffle')
    
    # 尝试获取 turtlebot3_gazebo 包路径
    try:
        tb3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    except:
        tb3_gazebo_dir = ''
    
    return LaunchDescription([
        
        # ============================================
        # 1. 启动 Gazebo Sim (gz sim)
        # ============================================
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', world_file],
            output='screen',
        ),
        
        # ============================================
        # 2. 生成 TurtleBot3 到场景中
        #    位置: (-4, -4, 0) 即场地左下角
        # ============================================
        # 注意: 你可能需要用自己的方式 spawn robot
        # 方式A: 用 ros2 run 命令
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'ros_gz_sim', 'create',
                '-name', 'waffle',
                '-topic', '/robot_description',
                '-x', '-4.0',
                '-y', '-4.0',
                '-z', '0.01',
            ],
            output='screen',
        ),
        
        # ============================================
        # 3. ros_gz_bridge: 桥接 Gazebo <-> ROS2 话题
        # ============================================
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/cmd_vel@geometry_msgs/msg/TwistStamped@gz.msgs.Twist',
                '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
                '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
                '/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
                '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
            ],
            output='screen',
        ),
        
        # ============================================
        # 4. Robot State Publisher (发布 TF)
        # ============================================
        # 如果你的 TurtleBot3 launch 已经包含这个，可以注释掉
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(
        #         os.path.join(tb3_gazebo_dir, 'launch', 'robot_state_publisher.launch.py')
        #     ),
        # ),
    ])
