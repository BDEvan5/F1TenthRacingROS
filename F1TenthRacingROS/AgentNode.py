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
import torch     
import yaml
import datetime

import numpy as np
from copy import copy

from F1TenthRacingROS.DriveNode import DriveNode
from matplotlib import pyplot as plt
from F1TenthRacingROS.TrackLine import TrackLine


NUM_BEAMS = 20
MAX_SPEED = 8
MAX_STEER = 0.4
N_WAYPOINTS_10 = 10
RANGE_FINDER_SCALE = 10
WAYPOINT_SCALE = 2.5
N_WAYPOINTS_20 = 20

def extract_scan(obs):
    scan = np.array(obs['scan']) 
    # scan = scan[180:-180] # remove behind fov
    inds = np.linspace(0, len(scan)-1, NUM_BEAMS).astype(int)
    scan = scan[inds]
    scan = np.clip(scan/RANGE_FINDER_SCALE, 0, 1)

    return scan

def extract_motion_variables(obs):
    speed = obs['state'][3] / MAX_SPEED
    # steering_angle = 0 #! problem...
    steering_angle = obs['state'][4] / MAX_STEER
    motion_variables = np.array([speed, steering_angle])
        
    return motion_variables

def extract_waypoints(obs, track, n_waypoints):
    pose = obs['state'][0:3]
    idx, dists = track.get_trackline_segment(pose[0:2])
    
    upcomings_inds = np.arange(idx, idx+n_waypoints)
    if idx + n_waypoints >= track.N:
        n_start_pts = idx + n_waypoints - track.N
        upcomings_inds[n_waypoints - n_start_pts:] = np.arange(0, n_start_pts)
        
    upcoming_pts = track.wpts[upcomings_inds]
    relative_pts = transform_waypoints(upcoming_pts, pose)
    relative_pts /= WAYPOINT_SCALE

    speeds = track.vs[upcomings_inds]
    scaled_speeds = speeds / MAX_SPEED

    return relative_pts, scaled_speeds


class EndArchitecture:
    def __init__(self, map_name, directory):
        self.action_space = 2
        self.state_space = NUM_BEAMS *2 + 1 

        self.n_scans = 2
        self.scan_buffer = np.zeros((self.n_scans, NUM_BEAMS))

    def process_observation(self, obs):
        scan = extract_scan(obs)
        if self.scan_buffer.all() ==0: # first reading
            for i in range(self.n_scans):
                self.scan_buffer[i, :] = scan 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan
        dual_scan = np.reshape(self.scan_buffer, (NUM_BEAMS * self.n_scans))

        speed = obs['state'][3] / MAX_SPEED
        nn_obs = np.concatenate((dual_scan, [speed]))

        return nn_obs


class PlanningArchitecture:
    def __init__(self, map_name, directory):
        self.waypoint_scale = 2.5
        self.state_space = N_WAYPOINTS_10 * 2 + 3 + NUM_BEAMS
        self.action_space = 2

        self.track = TrackLine(map_name, False, directory=directory)
    
    def process_observation(self, obs):
        relative_pts, _speeds = extract_waypoints(obs, self.track, N_WAYPOINTS_10)
        motion_variables = extract_motion_variables(obs)
        scan = extract_scan(obs)
        state = np.concatenate((scan, relative_pts.flatten(), motion_variables))
        
        return state

class TrajectoryArchitecture:
    def __init__(self, map_name, directory):
        self.state_space = N_WAYPOINTS_20 * 3 + 3
        self.waypoint_scale = 2.5

        self.action_space = 2
        self.track = TrackLine(map_name, True, directory=directory)
    
    def process_observation(self, obs):
        motion_variables = extract_motion_variables(obs)
        relative_pts, scaled_speeds = extract_waypoints(obs, self.track, N_WAYPOINTS_10)

        relative_pts = np.concatenate((relative_pts, scaled_speeds[:, None]), axis=-1)
        relative_pts = relative_pts.flatten()
        state = np.concatenate((relative_pts, motion_variables))
        
        return state


def transform_action(nn_action):
    steering_angle = nn_action[0] * MAX_STEER
    speed = (nn_action[1] + 1) * (MAX_SPEED  / 2 - 0.5) + 1
    speed = min(speed, MAX_SPEED) # cap the speed

    action = np.array([steering_angle, speed])

    return action
    
        
def transform_waypoints(wpts, pose):
    new_pts = wpts - pose[0:2]
    new_pts = new_pts @ np.array([[np.cos(pose[2]), -np.sin(pose[2])], [np.sin(pose[2]), np.cos(pose[2])]])
    
    return new_pts



architecture_dict = {"Game": PlanningArchitecture,
                     "TrajectoryFollower": TrajectoryArchitecture,
                     "endToEnd": EndArchitecture}
# architecture_dict = {"planning": PlanningArchitecture,
#                      "trajectory": TrajectoryArchitecture,
#                      "endToEnd": EndArchitecture}


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

        self.directory = self.params.directory
        self.agent = TestSAC(agent_name, self.directory + f"Data/PreTrained/{agent_name}/")
        architecture = agent_name.split("_")[2]
        self.architecture = architecture_dict[architecture](map_name, self.directory)

        self.speed_limit = self.params.speed_limit

    def calculate_action(self, observation):
        """
            Use the observation to calculate an action that is returned
        """
        
        nn_state = self.architecture.process_observation(observation)
        nn_action = self.agent.act(nn_state)
        action = transform_action(nn_action)

        # self.get_logger().info(f"nn_state: {nn_state}")
        current_steer = observation['state'][4]
        self.get_logger().info(f"Speed: {nn_state[-1]*8:.2f} --> SteerA: {action[0]:.4f} -> SteerC: {current_steer:.4f} ")

        action[0] = action[0]  #* 0.5
        action[1] = np.clip(action[1], 0, self.speed_limit)

        return action        
    
    def lap_complete_callback(self):
        self.send_drive_message([0, 0])
        run_path = self.experiment_history.save_experiment(self.params.agent_name)
        save_params = copy(self.params.__dict__)
        ct = datetime.datetime.now()
        save_params["time"] = f"{ct.month}_{ct.day}_{ct.hour}_{ct.minute}"
        with open(run_path + "RunParams.yaml", 'w') as f:
            yaml.dump(save_params, f)



def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    node.run_lap()
    rclpy.spin(node)

if __name__ == '__main__':
    main()

