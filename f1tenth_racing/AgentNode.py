from abc import abstractmethod
from argparse import Namespace
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseWithCovarianceStamped
import time
import math
from f1tenth_racing.DriveNode import DriveNode


import numpy as np
from copy import copy

from f1tenth_racing.Architectures import EndArchitecture, PlanningArchitecture, TrajectoryArchitecture

architecture_dict = {"planning": PlanningArchitecture,
                     "trajectory": TrajectoryArchitecture,
                     "endToEnd": EndArchitecture}


import torch     
class TestSAC:
    def __init__(self, filename, directory):
        self.actor = torch.load(directory + f'{filename}_actor.pth')

    def act(self, state):
        state = torch.FloatTensor(state)
        action, log_prob = self.actor(state)
        
        return action.detach().numpy()
      
class AgentNode(DriveNode):
    def __init__(self):
        super().__init__('nn_agent')
        
        agent_name = self.params.agent_name
        map_name = self.params.map_name
        architecture = self.params.architecture

        self.directory = self.params.directory
        self.agent = TestSAC(agent_name, self.directory + f"Data/PreTrained/{agent_name}/")
        self.architecture = architecture_dict[architecture](map_name)

        self.speed_limit = self.params.speed_limit

    def calculate_action(self, observation):
        """
            Use the observation to calculate an action that is returned
        """
        
        nn_state = self.architecture.process_observation(observation)
        nn_action = self.agent.act(nn_state)
        action = self.architecture.transform_action(nn_action)

        # self.get_logger().info(f"nn_state: {nn_state}")
        current_steer = observation['state'][4]
        self.get_logger().info(f"Speed: {nn_state[-1]*8:.2f} --> SteerA: {action[0]:.4f} -> SteerC: {current_steer:.4f} ")

        action[0] = action[0]  #* 0.5
        action[1] = np.clip(action[1], 0, self.speed_limit)

        return action        


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    node.run_lap()
    rclpy.spin(node)

if __name__ == '__main__':
    main()

