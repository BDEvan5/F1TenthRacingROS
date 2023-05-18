import numpy as np
from f1tenth_racing.TrackLine import TrackLine
from matplotlib import pyplot as plt



NUM_BEAMS = 20
MAX_SPEED = 8
MAX_STEER = 0.4
N_WAYPOINTS_10 = 10
RANGE_FINDER_SCALE = 10
WAYPOINT_SCALE = 2.5
N_WAYPOINTS_20 = 20



class EndArchitecture:
    def __init__(self, map_name):
        self.action_space = 2
        n_beams = 20
        self.state_space = n_beams + 1 

    def process_observation(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
        """
        scan = np.array(obs['scan']) 
        # scan = scan[180:-180] # remove behind fov
        inds = np.linspace(0, len(scan)-1, NUM_BEAMS).astype(int)
        scan = scan[inds]
        scan = scan - 0.2 # remove car from scan
        # scan = scan[::-1]
        speed = obs['state'][3] / MAX_SPEED
        scan = np.clip(scan/RANGE_FINDER_SCALE, 0, 1)
        nn_obs = np.concatenate((scan, [speed]))

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * MAX_STEER 
        speed = (nn_action[1] + 1) * (MAX_SPEED  / 2 - 0.5) + 1
        speed = min(speed, MAX_SPEED) # cap the speed

        action = np.array([steering_angle, speed])

        return action


class PlanningArchitecture:
    def __init__(self, map_name):
        self.waypoint_scale = 2.5
        self.state_space = N_WAYPOINTS_10 * 2 + 3 + NUM_BEAMS
        self.action_space = 2

        self.track = TrackLine(map_name, False)
    
    def process_observation(self, obs):
        pose = obs['state'][0:2]
        idx, dists = self.track.get_trackline_segment(pose)
        
        upcomings_inds = np.arange(idx, idx+N_WAYPOINTS_10)
        if idx + N_WAYPOINTS_10 >= self.track.N:
            n_start_pts = idx + N_WAYPOINTS_10 - self.track.N
            upcomings_inds[N_WAYPOINTS_10 - n_start_pts:] = np.arange(0, n_start_pts)
            
        upcoming_pts = self.track.wpts[upcomings_inds]
        
        relative_pts = transform_waypoints(upcoming_pts, pose, obs['state'][2])
        relative_pts /= WAYPOINT_SCALE
        relative_pts = relative_pts.flatten()
        
        speed = obs['state'][3] / MAX_SPEED
        # anglular_vel = obs['ang_vels_z'][0] / np.pi
        anglular_vel = 0 #! problem...
        steering_angle = obs['state'][4] / MAX_STEER
        
        scan = np.array(obs['scan'])
        inds = np.linspace(0, len(scan)-1, NUM_BEAMS).astype(int)
        scan = scan[inds]
        scan = np.clip(scan/RANGE_FINDER_SCALE, 0, 1)
        
        motion_variables = np.array([speed, anglular_vel, steering_angle])
        state = np.concatenate((scan, relative_pts.flatten(), motion_variables))
        
        return state
    
    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * MAX_STEER
        speed = (nn_action[1] + 1) * (MAX_SPEED  / 2 - 0.5) + 1
        speed = min(speed, MAX_SPEED) # cap the speed

        action = np.array([steering_angle, speed])
        self.previous_action = action

        return action

class TrajectoryArchitecture:
    def __init__(self, map_name):
        self.state_space = N_WAYPOINTS * 3 + 3
        self.waypoint_scale = 2.5

        self.action_space = 2
        self.track = TrackLine(map_name, True)
    
    def process_observation(self, obs):
        pose = obs['state'][0:2]
        idx, dists = self.track.get_trackline_segment(pose)
        
        speed = obs['state'][3] / MAX_SPEED
        # anglular_vel = obs['ang_vels_z'][0] / np.pi
        anglular_vel = 0 #! problem here....
        steering_angle = obs['state'][4] / MAX_STEER
        
        upcomings_inds = np.arange(idx+1, idx+N_WAYPOINTS_20+1)
        if idx + N_WAYPOINTS_20 + 1 >= self.track.N:
            n_start_pts = idx + N_WAYPOINTS_20 + 1 - self.track.N
            upcomings_inds[N_WAYPOINTS_20 - n_start_pts:] = np.arange(0, n_start_pts)
            
        upcoming_pts = self.track.wpts[upcomings_inds]
        
        relative_pts = transform_waypoints(upcoming_pts, pose, obs['state'][2])
        relative_pts /= WAYPOINT_SCALE
        
        speeds = self.track.vs[upcomings_inds]
        scaled_speeds = speeds / MAX_SPEED
        relative_pts = np.concatenate((relative_pts, scaled_speeds[:, None]), axis=-1)
        
        relative_pts = relative_pts.flatten()
        motion_variables = np.array([speed, anglular_vel, steering_angle])
        state = np.concatenate((relative_pts, motion_variables))
        
        return state
    
    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * MAX_STEER
        speed = (nn_action[1] + 1) * (MAX_SPEED  / 2 - 0.5) + 1
        speed = min(speed, MAX_SPEED) # cap the speed

        action = np.array([steering_angle, speed])
        self.previous_action = action

        return action
     
     
     
        
def transform_waypoints(wpts, position, orientation):
    new_pts = wpts - position
    new_pts = new_pts @ np.array([[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
    
    return new_pts
    
def plot_state(state):
    pts = np.reshape(state[:20], (10, 2))
    
    plt.figure(1)
    plt.clf()
    plt.plot(pts[:, 0], pts[:, 1], 'ro-')
    
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    plt.pause(0.00001)
    # plt.show()
    
    
    
