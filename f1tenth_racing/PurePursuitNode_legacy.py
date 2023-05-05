from f1tenth_racing.DriveNode import DriveNode

import rclpy
import time
import numpy as np

from f1tenth_racing.Trajectory import *


class PurePursuitNode(DriveNode):
    def __init__(self):
        super().__init__("pure_pursuit")

        self.declare_parameter('n_laps')
        self.declare_parameter('map_name')
        map_name = self.get_parameter('map_name').value
        self.get_logger().info(f"Map name: {map_name}")

        self.trajectory = Trajectory(map_name)

        self.lookahead = 1
        self.v_min_plan = 1
        self.wheelbase =  0.33
        self.max_steer = 0.4
        self.vehicle_speed = 2

        self.n_laps = self.get_parameter('n_laps').value
        self.get_logger().info(f"Number of laps to run: {self.n_laps}")

    def calculate_action(self, observation):
        state = observation['state']
        position = state[0:2]
        theta = state[2]
        lookahead_point = self.trajectory.get_current_waypoint(position, self.lookahead)

        speed, steering_angle = get_actuation(theta, lookahead_point, position, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        # action = np.array([steering_angle, speed])
        action = np.array([steering_angle, self.vehicle_speed])

        return action

    def lap_complete_callback(self):
        print(f"Lap complee: {self.current_lap_time}")


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


