from abc import abstractmethod
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseWithCovarianceStamped
import time
import math

from geometry_msgs.msg import TransformStamped
import tf2_ros

import numpy as np
from copy import copy
from argparse import Namespace

import os
import datetime
import csv
import yaml


def ensure_path_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


class ExperimentHistory:
    def __init__(self, results_directory):
        self.states = []
        self.actions = []
        self.scans = []
        self.results_directory = results_directory
        ensure_path_exists(self.results_directory)

    def add_data(self, state, action, scan):
        self.states.append(state)
        self.actions.append(action)
        self.scans.append(scan)

    def save_experiment(self, name, i=0):  
        path = self.results_directory + name + "/"
        ensure_path_exists(path)
        for i in range(100):
            run_path  = f"Run_{i}/"
            if os.path.exists(path + run_path): continue
            os.mkdir(path + run_path)
            break

        path = path + run_path
        name = f"Run_{i}"

        self.scans = np.array(self.scans)
        np.save(path + f"{name}_scans_{i}", self.scans)
        with open(path + f'{name}_states_{i}.csv', 'w') as f:
            csvwriter = csv.writer(f) 
            csvwriter.writerow(['X', "Y", "Theta", "Speed", "Steering"]) 
            csvwriter.writerows(self.states)
        with open(path + f'{name}_actions_{i}.csv', 'w') as f:
            csvwriter = csv.writer(f) 
            csvwriter.writerow(['Steering', "Speed"]) 
            csvwriter.writerows(self.actions)


        self.states = []
        self.actions = []
        self.scans = []

        return path 

        
def load_params_robust():
    try:
        d = "/home/benjy/sim_ws/src/F1TenthRacingROS/"
        with open(d + "config/params.yaml") as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
    except:
        d = "/home/jetson/f1tenth_ws/src/F1TenthRacingROS/"
        with open(d + "config/params.yaml") as file:
            params = yaml.load(file, Loader=yaml.FullLoader)

    params = Namespace(**params)
    return params


class DriveNode(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        
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
        self.lap_times = []

        self.params = load_params_robust()
        self.n_laps = self.params.n_laps

        simulation_time = 0.1
        
        qos_policy = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, 
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            # reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT, 
            history=rclpy.qos.HistoryPolicy.KEEP_LAST , depth=1)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', qos_profile=qos_policy)
        self.cmd_timer = self.create_timer(simulation_time, self.drive_callback)

        self.odom_subscriber = self.create_subscription(Odometry, self.params.odom_topic, self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.current_drive_sub = self.create_subscription(AckermannDrive, 'ego_racecar/current_drive', self.current_drive_callback, 10)

        self.ego_reset_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose', 10)

        self.experiment_history = ExperimentHistory(self.params.results_directory)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # self.delay_steps = self.params.delay_steps
        self.action_buffer = np.zeros((self.params.action_delay+1, 2))
        self.position_buffer = np.zeros((self.params.delay_steps+1, 3))
        # self.current_position_time = time.time()

        self.true_pos = np.zeros(2)
        self.true_theta = 0.0

    def broadcast_transform(self, pose):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'vehicle'
        transform.transform.translation.x = pose[0]
        transform.transform.translation.y = pose[1]
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = np.sin(pose[2]/2)
        transform.transform.rotation.w = np.cos(pose[2]/2)
        self.tf_broadcaster.sendTransform(transform)

    def current_drive_callback(self, msg):
        # self.steering_angle = 0
        pass
        # self.steering_angle = msg.steering_angle

    def odom_callback(self, msg):
        pos_x, pos_y = msg.pose.pose.position.x, msg.pose.pose.position.y
        x, y, z = quaternion_to_euler_angle(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        theta = z * np.pi / 180

        odom = np.array([pos_x, pos_y, theta])
        self.position_buffer = np.roll(self.position_buffer, -1, axis=0)
        self.position_buffer[-1] = odom
        current_odom = self.position_buffer[0]

        self.position = current_odom[:2]
        self.theta = current_odom[2]
        # self.velocity = current_odom[3]

        self.velocity = msg.twist.twist.linear.x

        self.true_pos = np.array([pos_x, pos_y])
        self.true_theta = theta


    # def odom_callback(self, msg):
    #     position = msg.pose.pose.position
    #     self.position = np.array([position.x, position.y])
    #     self.velocity = msg.twist.twist.linear.x

    #     x, y, z = quaternion_to_euler_angle(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
    #     theta = z * np.pi / 180
    #     self.theta = copy(theta)

        # self.get_logger().info(f"Position: {self.position} Theta: {self.theta} Velocity: {self.velocity}")

    def scan_callback(self, msg):
        self.scan = np.array(msg.ranges)

    def lap_done(self):
        self.current_lap_time = time.time() - self.lap_start_time
        self.get_logger().info(f"Run {self.lap_count} Complete in time: {self.current_lap_time}")
        self.lap_times.append(self.current_lap_time)
        self.lap_complete_callback()

        self.lap_count += 1

        if self.lap_count >= self.n_laps:
            self.running = False
            self.get_logger().info(f"Laps are completed: {self.lap_count} complete -> Running Set to: {self.running}")
            self.save_data_callback()
            self.ego_reset()
            self.destroy_node()

        self.current_lap_time = 0.0
        self.num_toggles = 0
        self.near_start = True
        self.toggle_list = 0
        self.lap_start_time = time.time()

    def drive_callback(self):
        if not self.running:
            return

        if self.check_lap_done(self.true_pos):
        # if self.check_lap_done(self.position):
        # if self.check_lap_done_full_map(self.position):
            self.lap_done()
            return
        
        observation = self.build_observation()

        action = self.calculate_action(observation)
        # self.steering_angle = 0.5 * self.steering_angle + 0.5* self.action_buffer[0, 0]
        self.steering_angle = 0.5 * self.steering_angle + 0.5* action[0]

        state = observation['state']
        self.experiment_history.add_data(state, action, self.scan)

        self.action_buffer = np.roll(self.action_buffer, -1, axis=0)
        self.action_buffer[-1] = action
        publish_action = self.action_buffer[0]

        self.send_drive_message(publish_action)

    @abstractmethod
    def calculate_action(self, observation):
        """
            Use the observation to calculate an action that is returned
        """
        raise NotImplementedError

    @abstractmethod
    def save_data_callback(self):
        lap_times = np.array(self.lap_times)
        self.get_logger().info(f"Saved lap times: {lap_times}")


    @abstractmethod
    def lap_complete_callback(self):
        pass

    # def send_drive_message(self, action):
    #     drive_msg = AckermannDrive()
    #     drive_msg.speed = float(action[1])
    #     drive_msg.steering_angle = float(action[0])
    #     self.drive_publisher.publish(drive_msg)
    
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
        self.broadcast_transform(state[:3])
        observation['state'] = state
        observation['reward'] = 0.0

        return observation

    def check_lap_done_full_map(self, position):
        start_x = 0
        start_y = 0 
        start_theta = 0
        start_rot = np.array([[np.cos(-start_theta), -np.sin(-start_theta)], [np.sin(-start_theta), np.cos(-start_theta)]])

        poses_x = np.array(position[0])-start_x
        poses_y = np.array(position[1])-start_y
        delta_pt = np.dot(start_rot, np.stack((poses_x, poses_y), axis=0))

        dist2 = delta_pt[0]**2 + delta_pt[1]**2
        closes = dist2 <= 1
        self.get_logger().info(f"checking done...: {position}")

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
    
    def check_lap_done(self, position):
        """
            Check if the car has completed a lap
        """
        # if position[1] < -10.5:
        if position[1] < -16.5:
            print(f"Lap completed.....")
            return True

    def ego_reset(self):
        msg = PoseWithCovarianceStamped() 

        msg.pose.pose.position.x = 0.0 
        msg.pose.pose.position.y = 0.0

        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0
        self.broadcast_transform(np.zeros(3))

        self.ego_reset_pub.publish(msg)

        self.get_logger().info("Finished Resetting Vehicle")

    def run_lap(self):
        time.sleep(0.1)
        self.ego_reset()
        time.sleep(0.1)

        self.current_lap_time = 0.0
        self.running = True

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

if __name__ == '__main__':
    pass

