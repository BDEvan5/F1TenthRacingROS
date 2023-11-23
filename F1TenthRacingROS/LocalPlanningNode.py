from argparse import Namespace
import rclpy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import yaml
from numba import njit
import numpy as np
from copy import copy
import datetime
import tf2_geometry_msgs

from F1TenthRacingROS.DriveNode import DriveNode
from F1TenthRacingROS.LocalMapGenerator import LocalMapGenerator
from F1TenthRacingROS.local_opt_min_curv import local_opt_min_curv

from scipy import interpolate, spatial, optimize
import trajectory_planning_helpers as tph


MAX_SPEED = 8
MAX_STEER = 0.4
WHEELBASE = 0.33
      

class LocalPlanningNode(DriveNode):
    def __init__(self):
        super().__init__('local_node')
        self.agent_name = "LocalPlanning"
        self.directory = self.params.directory
        self.speed_limit = self.params.speed_limit
        self.lookahead = self.params.lookahead

        self.local_map_generator = LocalMapGenerator()
        self.local_track = None
        self.raceline = None
        self.s_raceline = None
        self.vs = None
        self.planner_params = load_parameter_file("LocalPlanningParams")

        p = self.planner_params
        self.ggv = np.array([[0, p.max_longitudinal_acc, p.max_lateral_acc],
                    [self.planner_params.max_speed, p.max_longitudinal_acc, p.max_lateral_acc]])
        self.ax_max_machine = np.array([[0, p.max_longitudinal_acc],
                                        [self.planner_params.max_speed, p.max_longitudinal_acc]])
        qos_policy = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, 
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            # reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT, 
            history=rclpy.qos.HistoryPolicy.KEEP_LAST , depth=1)
        self.local_map_publisher = self.create_publisher(Path, '/local_track', qos_policy)
        self.ego_reset()


    def calculate_action(self, observation):
        """
            Use the observation to calculate an action that is returned
        """
        self.broadcast_transform(observation['state'][:3])
        scan = observation['scan']
        if np.all(scan == 0):
            return np.zeros(2)
        self.local_track = self.local_map_generator.generate_line_local_map(scan)
        if len(self.local_track) < 4:
            return np.array([0, 6])
            # return np.zeros(2)
        if len(self.local_track) > 25:
            self.local_track = self.local_track[:25]
        
        print(f"Local track length: {len(self.local_track)}")
        self.publish_local_track()

        self.generate_minimum_curvature_path()
        self.generate_max_speed_profile()

        action = self.pure_pursuit_racing_line(observation['state'][3])

        action[0] = np.clip(action[0], -MAX_STEER, MAX_STEER)
        action[1] = np.clip(action[1], 0, self.speed_limit)

        return action        

    def publish_local_track(self):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "vehicle"
        for point in self.local_track:
            pose = PoseStamped()
            pose.header.frame_id = "vehicle"
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            msg.poses.append(pose)
        self.local_map_publisher.publish(msg)

    def lap_complete_callback(self):
        self.send_drive_message([0, 0])
        run_path = self.experiment_history.save_experiment(self.agent_name)
        save_params = copy(self.params.__dict__)
        ct = datetime.datetime.now()
        save_params["time"] = f"{ct.month}_{ct.day}_{ct.hour}_{ct.minute}"
        with open(run_path + "RunParams.yaml", 'w') as f:
            yaml.dump(save_params, f)

    def generate_minimum_curvature_path(self):
        track = self.local_track.copy()

        track[:, 2:] -= self.planner_params.path_exclusion_width / 2

        try:
            alpha, nvecs = local_opt_min_curv(track, self.planner_params.kappa_bound, 0, fix_s=True, fix_e=False)
            self.raceline = track[:, :2] + np.expand_dims(alpha, 1) * nvecs
        except Exception as e:
            self.raceline = track[:, :2]

        self.tck = interpolate.splprep([self.raceline[:, 0], self.raceline[:, 1]], k=3, s=0)[0]
        
    def generate_max_speed_profile(self):
        max_speed = self.planner_params.max_speed
        mu = self.planner_params.mu * np.ones_like(self.raceline[:, 0])

        raceline_el_lengths = np.linalg.norm(np.diff(self.raceline, axis=0), axis=1)
        self.s_raceline = np.insert(np.cumsum(raceline_el_lengths), 0, 0)
        _, raceline_curvature = tph.calc_head_curv_num.calc_head_curv_num(self.raceline, raceline_el_lengths, False)

        self.vs = tph.calc_vel_profile.calc_vel_profile(self.ax_max_machine, raceline_curvature, raceline_el_lengths, False, 0, self.planner_params.vehicle_mass, ggv=self.ggv, mu=mu, v_max=max_speed, v_start=max_speed, v_end=max_speed)

    def calculate_zero_point_progress(self):
        n_pts = np.count_nonzero(self.s_raceline < 5) # search first 4 m
        s_raceline = self.s_raceline[:n_pts]
        raceline = self.raceline[:n_pts]
        new_points = np.linspace(0, s_raceline[-1], int(s_raceline[-1]*100)) #cm accuracy
        xs, ys = interp_2d_points(new_points, s_raceline, raceline)
        raceline = np.concatenate([xs[:, None], ys[:, None]], axis=-1)
        dists = np.linalg.norm(raceline, axis=1)
        t_new = (new_points[np.argmin(dists)] / self.s_raceline[-1])

        return [t_new]

    def pure_pursuit_racing_line(self, vehicle_speed):
        lookahead_distance = self.planner_params.constant_lookahead + (vehicle_speed/self.planner_params.max_speed) * (self.planner_params.variable_lookahead)
        current_s = self.calculate_zero_point_progress()
        lookahead_s = current_s + lookahead_distance / self.s_raceline[-1]
        lookahead_point = np.array(interpolate.splev(lookahead_s, self.tck, ext=3)).T
        if len(lookahead_point.shape) > 1: lookahead_point = lookahead_point[0]

        exact_lookahead = np.linalg.norm(lookahead_point)
        steering_angle = get_local_steering_actuation(lookahead_point, exact_lookahead, self.planner_params.wheelbase) 
        speed = np.interp(current_s, self.s_raceline/self.s_raceline[-1], self.vs)[0]

        return np.array([steering_angle, speed])



@njit(fastmath=False, cache=True)
def get_local_steering_actuation(lookahead_point, lookahead_distance, wheelbase):
    waypoint_y = lookahead_point[1]
    if np.abs(waypoint_y) < 1e-6:
        return 0.0
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return steering_angle
   
def interp_2d_points(ss, xp, points):
    xs = np.interp(ss, xp, points[:, 0])
    ys = np.interp(ss, xp, points[:, 1])
    
    return xs, ys


def load_parameter_file(planner_name):
    file_name = f"config/{planner_name}.yaml"
    with open(file_name, 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return Namespace(**params)

def main(args=None):
    rclpy.init(args=args)
    node = LocalPlanningNode()
    node.run_lap()
    rclpy.spin(node)

if __name__ == '__main__':
    main()

