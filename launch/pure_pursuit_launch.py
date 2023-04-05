
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
        "/home/benjy/sim_ws/src/f1tenth_racing/",
        'config',
        'testing_params.yaml'
    )

    testing_node = Node(
        package='f1tenth_racing',
        executable='pure_pursuit',
        name='pure_pursuit',
        parameters=[config]
    )


    # localize_config = os.path.join(
    #         get_package_share_directory('particle_filter'),
    #         'config',
    #         'localize.yaml'
    #     )
    # localize_config_dict = yaml.safe_load(open(localize_config, 'r'))
    # map_name = localize_config_dict['map_server']['ros__parameters']['map']
    # localize_la = DeclareLaunchArgument(
    #     'localize_config',
    #     default_value=localize_config,
    #     description='Localization configs')

    # nodes
    # pf_node = Node(
    #     package='particle_filter',
    #     executable='particle_filter',
    #     name='particle_filter',
    #     parameters=[localize_config]
    #     # parameters=[LaunchConfiguration('localize_config')]
    # )

    ld.add_action(testing_node)
    # ld.add_action(pf_node)

    return ld

