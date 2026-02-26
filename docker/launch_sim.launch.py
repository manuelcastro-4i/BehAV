import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
import xacro


def generate_launch_description():

    package_name = 'gazebo_sim'
    namespace = '/robot1'
    name = 'robot1'

    pkg_path = os.path.join(get_package_share_directory(package_name))
    xacro_file = os.path.join(pkg_path, 'xacro', 'robot.xacro')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    declare_use_sim_time = DeclareLaunchArgument(
        name='use_sim_time', default_value=use_sim_time,
        description='Use simulation time'
    )

    world = os.path.join(pkg_path, 'world', 'empty.world')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments={
            'gz_args': ['-r -v4 ', world],
            'on_exit_shutdown': 'true'
        }.items()
    )

    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        namespace=namespace,
        arguments=[
            '-topic', f'{namespace}/robot_description',
            '-name', f'{namespace}/my_bot',
            '-z', '0.4'
        ],
        output='screen'
    )

    robot_desc = xacro.process_file(
        xacro_file,
        mappings={'robot_name': name}
    ).toxml()

    params = {'robot_description': robot_desc, 'use_sim_time': use_sim_time}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        namespace=namespace,
        parameters=[params],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")]
    )

    # Clock bridge only (minimal, stable)
    bridge_params = os.path.join(get_package_share_directory(package_name), 'config', 'gz_bridge.yaml')
    ros_gz_bridge_clock = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=['--ros-args', '-p', f'config_file:={bridge_params}']
    )

    # BehAV bridge: odom + camera + scan + world_poses (started by sim_adapters.py,
    # but also available here if needed for debugging). sim_adapters.py uses
    # gz_pose_bridge.yaml which is mounted at /app/gz_pose_bridge.yaml.
    # We intentionally do NOT start an additional bridge here to avoid duplicate publishers.

    # Intentionally removed (causes crashes or depends on gz_ros2_control which is not installed):
    #   - ros_gz_bridge for IMU/scan/TF/joint_states (SIGABRT with corrupted double-linked list)
    #   - joint_state_broadcaster spawner (controller_manager never available)
    #   - joint_group_controller spawner (same)
    #   - robot_controller_gazebo.py (depends on controllers)
    #   - cmd_vel_pub.py (crashes without controllers)
    #   - QuadrupedOdometryNode.py (depends on above)
    #   - rviz2 (SIGABRT on headless display)

    return LaunchDescription([
        declare_use_sim_time,
        node_robot_state_publisher,
        gazebo,
        spawn_entity,
        ros_gz_bridge_clock,
    ])
