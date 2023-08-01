
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os
import yaml
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    ld = LaunchDescription()

    config = os.path.join(
        "/home/benjy/sim_ws/src/F1TenthRacingROS/",
        'config',
        'testing_params.yaml'
    )

    testing_node = Node(
        package='F1TenthRacingROS',
        executable='nn_agent',
        name='nn_agent',
        parameters=[config]
    )

    ld.add_action(testing_node)

    return ld

