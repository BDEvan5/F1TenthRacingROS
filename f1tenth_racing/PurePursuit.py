from argparse import Namespace
import rclpy
import yaml
from numba import njit
import numpy as np
from copy import copy

from f1tenth_racing.DriveNode import DriveNode
from matplotlib import pyplot as plt
from f1tenth_racing.TrackLine import TrackLine


MAX_SPEED = 8
MAX_STEER = 0.4
WHEELBASE = 0.33
import datetime
      

class PurePursuitNode(DriveNode):
    def __init__(self):
        super().__init__('pp_node')
        
        architecture = self.params.architecture
        assert architecture == "pure_pursuit", f"Incorrect architecture {architecture}"

        self.agent_name = self.params.agent_name
        map_name = self.params.map_name
        self.trajectory = TrackLine(map_name, True)

        self.directory = self.params.directory
        self.speed_limit = self.params.speed_limit
        self.lookahead = self.params.lookahead

    def calculate_action(self, observation):
        """
            Use the observation to calculate an action that is returned
        """
        state = observation['state']
        position = state[0:2]
        theta = state[2]
        lookahead_point = self.trajectory.get_lookahead_point(position, self.lookahead)

        speed, steering_angle = get_actuation(theta, lookahead_point, position, self.lookahead, WHEELBASE)
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)

        action = np.array([steering_angle, speed])
        action[1] = np.clip(action[1], 0, self.speed_limit)

        return action        
    
    def lap_complete_callback(self):
        self.send_drive_message([0, 0])
        run_path = self.experiment_history.save_experiment(self.agent_name)
        save_params = copy(self.params.__dict__)
        ct = datetime.datetime.now()
        save_params["time"] = f"{ct.month}_{ct.day}_{ct.hour}_{ct.minute}"
        with open(run_path + "RunParams.yaml", 'w') as f:
            yaml.dump(save_params, f)


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    node.run_lap()
    rclpy.spin(node)

if __name__ == '__main__':
    main()

