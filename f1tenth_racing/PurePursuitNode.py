import time
import numpy as np

from abc import abstractmethod
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseWithCovarianceStamped
import time, math

from copy import copy

from f1tenth_racing.Trajectory import *
from f1tenth_racing.TrackLine import *


def quaternion_to_euler_angle(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z

class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit')
        
        # abstract variables
        self.planner = None
        self.supervision = False 
        self.supervisor = None

        # current vehicle state
        self.position = np.array([0, 0])
        self.velocity = 0
        self.theta = 0
        self.steering_angle = 0.0
        self.scan = None # move to param file

        self.lap_counts = 0
        self.toggle_list = 0
        self.near_start = True
        self.lap_start_time = time.time()
        self.current_lap_time = 0.0
        self.running = False

        self.lap_count = 0 
        self.n_laps = None
        self.lap_times = []

        self.logger = None

        simulation_time = 0.1
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.cmd_timer = self.create_timer(simulation_time, self.drive_callback)

        self.odom_subscriber = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)

        self.current_drive_sub = self.create_subscription(AckermannDrive, 'ego_racecar/current_drive', self.current_drive_callback, 10)

        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        
        self.declare_parameter('n_laps')
        self.declare_parameter('map_name')
        map_name = self.get_parameter('map_name').value
        self.get_logger().info(f"Map name: {map_name}")

        self.trajectory = TrackLine(map_name, True)
        # self.trajectory = Trajectory(map_name)

        self.lookahead = 1.4
        self.v_min_plan = 1
        self.wheelbase =  0.33
        self.max_steer = 0.4
        self.vehicle_speed = 2

        self.n_laps = self.get_parameter('n_laps').value
        self.get_logger().info(f"Number of laps to run: {self.n_laps}")


        self.ego_reset_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose', 10)

    def current_drive_callback(self, msg):
        self.steering_angle = msg.steering_angle

    def odom_pf_callback(self, msg):
        position = msg.pose.pose.position
        pf_position = np.array([position.x, position.y])
        pf_velocity = msg.twist.twist.linear.x

        x, y, z = quaternion_to_euler_angle(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        pf_theta = z * np.pi / 180
        
        self.get_logger().info(f"PF Position: {pf_position} -- TruePos: {self.position} -> diff: {self.position - pf_position}")
        
    def odom_callback(self, msg):
        position = msg.pose.pose.position
        self.position = np.array([position.x, position.y])
        self.velocity = msg.twist.twist.linear.x

        x, y, z = quaternion_to_euler_angle(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        theta = z * np.pi / 180
        self.theta = copy(theta)

    def scan_callback(self, msg):
        self.scan = np.array(msg.ranges)

    def lap_done(self):
        self.current_lap_time = time.time() - self.lap_start_time
        self.get_logger().info(f"Run {self.lap_count} Complete in time: {self.current_lap_time}")
        self.lap_times.append(self.current_lap_time)
        self.lap_complete_callback()

        self.lap_count += 1

        if self.lap_count == self.n_laps:
            self.running = False
            self.save_data_callback()
            self.ego_reset()
            self.destroy_node()

        if self.logger: self.logger.reset_logging()

        self.current_lap_time = 0.0
        self.num_toggles = 0
        self.near_start = True
        self.toggle_list = 0
        self.lap_start_time = time.time()
        
        # self.ego_reset() # for debuggin the pf

    def drive_callback(self):
        if not self.running:
            return

        if self.check_lap_done(self.position):
            self.lap_done()
        
        observation = self.build_observation()

        action = self.calculate_action(observation)

        self.send_drive_message(action)

    @abstractmethod
    def save_data_callback(self):
        lap_times = np.array(self.lap_times)
        self.get_logger().info(f"Saved lap times: {lap_times}")


    def calculate_action(self, observation):
        state = observation['state']
        position = state[0:2]
        theta = state[2]
        lookahead_point = self.trajectory.get_lookahead_point(position, self.lookahead)

        speed, steering_angle = get_actuation(theta, lookahead_point, position, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        # action = np.array([steering_angle, speed * 0.9])
        action = np.array([steering_angle, speed * 0.8])
        # action = np.array([steering_angle, 0.5])
        # action = np.array([steering_angle, self.vehicle_speed])

        return action

    def lap_complete_callback(self):
        print(f"Lap complee: {self.current_lap_time}")

    def send_drive_message(self, action):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(action[1])
        drive_msg.drive.steering_angle = float(action[0])
        self.drive_publisher.publish(drive_msg)

    def build_observation(self):
        """
        Observation:
            scan: LiDAR scan 
            state: [pose_x, pose_y, theta, velocity, steering angle]
            reward: 0 - created here to store the reward later
        
        """
        observation = {}
        observation["scan"] = self.scan
        if observation["scan"] is None: observation["scan"] = np.zeros(1080)

        state = np.array([self.position[0], self.position[1], self.theta, self.velocity, self.steering_angle])
        observation['state'] = state
        observation['reward'] = 0.0

        return observation

    def check_lap_done(self, position):
        start_x = 0
        start_y = 0 
        start_theta = 0
        start_rot = np.array([[np.cos(-start_theta), -np.sin(-start_theta)], [np.sin(-start_theta), np.cos(-start_theta)]])

        poses_x = np.array(position[0])-start_x
        poses_y = np.array(position[1])-start_y
        delta_pt = np.dot(start_rot, np.stack((poses_x, poses_y), axis=0))

        dist2 = delta_pt[0]**2 + delta_pt[1]**2
        closes = dist2 <= 1
        if closes and not self.near_start:
            self.near_start = True
            self.toggle_list += 1
            self.get_logger().info(f"Near start true: {position}")
        elif not closes and self.near_start:
            self.near_start = False
            self.toggle_list += 1
            self.get_logger().info(f"Near start false: {position}")
            # print(self.toggle_list)
        self.lap_counts = self.toggle_list // 2
        
        done = self.toggle_list >= 2
        
        return done

    def ego_reset(self):
        msg = PoseWithCovarianceStamped() 

        msg.pose.pose.position.x = 0.0 
        msg.pose.pose.position.y = 0.0

        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        # msg.pose.pose.orientation.z = -0.707
        # msg.pose.pose.orientation.w = 0.707
        # msg.pose.pose.orientation.x = 0.0
        # msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0
        # msg.pose.pose.orientation.z = 1.0
        # msg.pose.pose.orientation.w = 0.0

        self.ego_reset_pub.publish(msg)

        self.get_logger().info("Finished Resetting: angle 180")

    def run_lap(self):
        time.sleep(0.1)
        self.ego_reset()
        time.sleep(0.1)

        self.current_lap_time = 0.0
        self.running = True



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


