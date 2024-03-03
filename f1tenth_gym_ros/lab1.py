import rclpy
from rclpy.node import Node
import sys
import time
import os

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import TransformBroadcaster

from ament_index_python.packages import get_package_share_directory

import gym
import numpy as np
from transforms3d import euler

# this node should get the current pose from odom and get a reference trajectory from a yaml file
# and publish ackermann drive commands to the car based on one of 4 controllers selected with a parameter
# the controllers are PID, Pure Pursuit, iLQR, and an optimal controller 
class Lab1(Node):
    def __init__(self, controller_type: str = 'pid'):
        super().__init__('lab1')
        self.get_logger().info("Lab 1 Node has been started")

        # get parameters
        self.controller = self.declare_parameter('controller', controller_type).value
        self.get_logger().info("Controller: " + self.controller)
        # to set the parameter, run the following command in a terminal when running a different controller
        # ros2 run f1tenth_gym_ros lab1.py --ros-args -p controller:=<controller_type>

        # get the current pose
        self.get_logger().info("Subscribing to Odometry")
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.odom_sub # prevent unused variable warning
        
        self.get_logger().info("Publishing to Ackermann Drive")
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.cmd_pub # prevent unused variable warning
        
        # get the reference trajectory
        self.get_logger().info("Loading Reference Trajectory")
        self.ref_traj = np.load(os.path.join(get_package_share_directory('f1tenth_gym_ros'),
                                            'resource',
                                            'ref_traj.npy'))
        self.ref_traj # prevent unused variable warning
        
        # create a timer to publish the control input every 20ms
        self.get_logger().info("Creating Timer")
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.timer # prevent unused variable warning
        
        self.pose = np.zeros(3)
        
        self.steering_angle = 0
        self.speed = 0
        self.theta_error = 0
        self.sum_cross_track_error = 0
        self.sum_along_track_error = 0
        self.deriv_cross_track_error = 0
        self.deriv_along_track_error = 0
        self.time = time.time()
        
        self.current_cross_track_error = 0
        self.current_along_track_error = 0
        self.cross_track_accumulated_error = 0
        self.along_track_accumulated_error = 0
        self.waypoint_index = 0
        
        self.moved = False
    
    def get_ref_pos(self):
        # get the next waypoint in the reference trajectory based on the current time
        waypoint = self.ref_traj[self.waypoint_index % len(self.ref_traj)]
        self.waypoint_index += 1
        return waypoint

    def log_accumulated_error(self):
        ref_pos = self.get_ref_pos()
        
        cross_track_error = 0
        along_track_error = 0
        # compute the cross track and along track error depending on the current pose and the next waypoint
        #### YOUR CODE HERE ####
        
        next_ref_pos = self.get_ref_pos()
        self.waypoint_index -= 1 # revert waypoint index
        
        x_ref = ref_pos[0]
        y_ref = ref_pos[1]
        next_x_ref = next_ref_pos[0]
        next_y_ref = next_ref_pos[1]
        x_pos = self.pose[0]
        y_pos = self.pose[1]
        
        theta_ref = np.arctan2(next_y_ref-y_ref, next_x_ref-x_ref)
        
        cross_track_error = -np.sin(theta_ref)*(x_pos-x_ref) + np.cos(theta_ref)*(y_pos-y_ref)
        along_track_error =  np.cos(theta_ref)*(x_pos-x_ref) + np.sin(theta_ref)*(y_pos-y_ref)
        
        self.sum_cross_track_error += cross_track_error
        self.sum_along_track_error += along_track_error
        
        self.theta_error = self.pose[2] - theta_ref
        
        self.deriv_cross_track_error = self.speed * np.sin(self.theta_error)
        self.deriv_along_track_error = self.speed * np.cos(self.theta_error)
        
        #### END OF YOUR CODE ####
        self.current_cross_track_error = cross_track_error
        self.current_along_track_error = along_track_error
        
        # log the accumulated error to screen and internally to be printed at the end of the run
        self.get_logger().info("Cross Track Error: " + str(cross_track_error))
        self.get_logger().info("Along Track Error: " + str(along_track_error))
        self.get_logger().info("Current Position(x): " + str(self.pose[0]))
        self.get_logger().info("Current Position(y): " + str(self.pose[1]))
        self.cross_track_accumulated_error += abs(cross_track_error)
        self.along_track_accumulated_error += abs(along_track_error)
        
    def odom_callback(self, msg):
        # get the current pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler.quat2euler([q.w, q.x, q.y, q.z])
        
        if not self.moved and (x < -1 and y > 3):
            self.moved = True
        elif self.moved and x > 0:
            raise EndLap
            
        
        self.pose = np.array([x, y, yaw])
        
    def timer_callback(self):
        self.log_accumulated_error()
        
        # compute the control input
        if self.controller == "pid_unicycle":
            u = self.pid_unicycle_control(self.pose)
        elif self.controller == "pid":
            u = self.pid_control(self.pose)
        elif self.controller == "pure_pursuit":
            u = self.pure_pursuit_control(self.pose)
        elif self.controller == "ilqr":
            u = self.ilqr_control(self.pose)
        elif self.controller == "optimal":
            u = self.optimal_control(self.pose)
        else:
            self.get_logger().info("Unknown controller")
            return
        
        # publish the control input
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.drive.steering_angle = u[0]
        cmd.drive.speed = u[1]
        self.cmd_pub.publish(cmd)

    def pid_control(self, pose):
        #### YOUR CODE HERE ####
       
        K_p = np.array([[0, 0.3], [1, 0]])
        K_i = np.array([[0, 0.005], [1, 0.01]])
        K_d = np.array([[0, 1], [0, 0]])
        
        total_error = np.array([self.current_along_track_error, self.current_cross_track_error])
        total_error = total_error.reshape(2, 1)
        total_integral_error = np.array([self.sum_along_track_error, self.sum_cross_track_error])
        total_integral_error = total_integral_error.reshape(2, 1)
        total_derivative_error = np.array([self.deriv_along_track_error, self.deriv_cross_track_error])
        total_derivative_error = total_derivative_error.reshape(2, 1)
        
        u = - (K_p @ total_error + K_i @ total_integral_error + K_d @ total_derivative_error)
        
        self.steering_angle = float(u[0])
        self.speed = float(u[1])
        
        return np.array([self.steering_angle, self.speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError
    
    def pid_unicycle_control(self, pose):
        #### YOUR CODE HERE ####
        
        K_p = np.array([[0, 0.3], [1, 0]])
        K_i = np.array([[0, 0.005], [1, 0.01]])
        K_d = np.array([[0, 1], [0, 0]])
        
        total_error = np.array([self.current_along_track_error, self.current_cross_track_error])
        total_error = total_error.reshape(2, 1)
        total_integral_error = np.array([self.sum_along_track_error, self.sum_cross_track_error])
        total_integral_error = total_integral_error.reshape(2, 1)
        total_derivative_error = np.array([self.deriv_along_track_error, self.deriv_cross_track_error])
        total_derivative_error = total_derivative_error.reshape(2, 1)
        
        u = - (K_p @ total_error + K_i @ total_integral_error + K_d @ total_derivative_error)
        
        self.steering_angle = float(u[0])
        self.speed = float(u[1])
        
        return np.array([self.steering_angle, self.speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError
    
    def pure_pursuit_control(self, pose):
        #### YOUR CODE HERE ####
        
        speed_parameter = 10
        look_ahead = np.ceil(self.speed*speed_parameter)
        
        K_p = np.array([[0, 0.3], [0.5, 0]])
        K_i = np.array([[0, 0.005], [0.1, 0.01]])
        K_d = np.array([[0, 1], [0, 0]])
        
        ref_pos = self.ref_traj[(self.waypoint_index + int(look_ahead) - 1) % len(self.ref_traj)]
        dist_to_ref_point = np.linalg.norm(ref_pos - pose[:2])
        
        alpha = pose[2] - np.arctan2(ref_pos[1]-pose[1], ref_pos[0]-pose[0])
        
        self.steering_angle = -np.arctan2(2*np.sin(alpha), dist_to_ref_point)
        
        total_error = np.array([self.current_along_track_error, self.current_cross_track_error])
        total_error = total_error.reshape(2, 1)
        total_integral_error = np.array([self.sum_along_track_error, self.sum_cross_track_error])
        total_integral_error = total_integral_error.reshape(2, 1)
        total_derivative_error = np.array([self.deriv_along_track_error, self.deriv_cross_track_error])
        total_derivative_error = total_derivative_error.reshape(2, 1)
        
        u = - (K_p @ total_error + K_i @ total_integral_error + K_d @ total_derivative_error)
        
        self.speed = float(u[1])
        
        return np.array([self.steering_angle, self.speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError
        
    def ilqr_control(self, pose):
        #### YOUR CODE HERE ####
        
        # Parameters
        max_iter = 2000
        look_ahead = 4
        num_states = 3
        num_actions = 2
        if (self.waypoint_index + look_ahead) > self.ref_traj.shape[0]:
            path_len = self.ref_traj.shape[0] - self.waypoint_index
        else:
            path_len = look_ahead
        d = 0.0381 # wheel length
        r = 0.0508 # wheel radius
        w = 0.3302 # wheelbase
        Q_f = np.eye(3)
        Q = np.eye(3)
        Q[2, 2] = 0.01
        R = np.eye(2) * 200
        rho = 80 * np.eye(3) # normalisation if needed
        
        # calculate real dt
        temp_time = time.time()
        dt = temp_time - self.time
        self.time = temp_time
        
        if path_len > 0:
            # reinitialise reference trajectory to include angle
            x_ref_traj = np.zeros((path_len, num_states))
            x_ref_traj[:, :2] = self.ref_traj[self.waypoint_index:(self.waypoint_index + path_len)]
            for idx in range(path_len-1):
                x_ref, y_ref = x_ref_traj[idx, :2]
                next_x_ref, next_y_ref = x_ref_traj[idx+1, :2]
                x_ref_traj[idx, 2] = np.arctan2(next_y_ref-y_ref, next_x_ref-x_ref)
        
            x_ref_traj[-1,2] = 0
        
            # Initialise control inputs
            U_init = np.ones((path_len, num_actions)) * 0.25 * (1 / dt)
            U_init[:, 0] = np.zeros((path_len)) / 1000
            U = np.copy(U_init)
            
            for _ in range(max_iter):
                # initialisation - find path using current control
                x_ref, y_ref, theta_ref = x_ref_traj[0]
                X = np.zeros((path_len, num_states))
                X[0] = pose
                for t in range(path_len - 1):
                    x, y, theta = X[t]
                    steering_speed, speed = U[t]
                    x_next = x + (speed * np.cos(theta) * dt)
                    y_next = y + (speed * np.sin(theta) * dt)
                    theta_next = theta + (speed * np.tan(steering_speed) * dt)
                    X[t + 1] = np.array([x_next, y_next, theta_next])
                    
                # backward pass
                u_T = np.zeros((num_actions, 1))
                x_T = X[-1]
                V_x = np.zeros((path_len, num_states))
                V_xx = np.zeros((path_len, num_states, num_states))
                V_x[-1] = Q_f @ (x_T - x_ref_traj[-1])
                V_xx[-1] = Q_f
                K = np.zeros((path_len - 1, num_actions, num_states,))
                d = np.zeros((path_len - 1, num_actions))
                
                for t in range(path_len - 2, -1, -1):
                    x_t = X[t]
                    u_t = U[t]
                    x, y, theta = x_t
                    steering_speed, speed = u_t
                    
                    # linearise dynamics arround current state & control
                    A = np.eye(num_states)
                    A[0, 2] = -speed * np.sin(theta) * dt
                    A[1, 2] =  speed * np.cos(theta) * dt
                    B = np.zeros((num_states, num_actions))
                    B[2, 0] = dt * speed / (w * np.cos(steering_speed) * np.cos(steering_speed))
                    B[0, 1] = dt * np.cos(theta)
                    B[1, 1] = dt * np.sin(theta)
                    B[2, 1] = dt * np.tan(steering_speed) / w
                    
                    # compute value function hessians and gradients
                    Q_x = Q @ x_t
                    Q_xx = Q
                    
                    V_x[t] = Q_x + (A.T @ V_x[t+1])
                    V_xx[t] = Q_xx + (A.T @ V_xx[t+1] @ A)
                    
                    Q_u = (R @ u_t) + (B.T @ V_x[t+1])
                    Q_ux = B.T @ (V_xx[t+1] + rho) @ A
                    Q_uu = R + (B.T @ (V_xx[t+1] + rho) @ B)
                    
                    Q_uu_inv = np.linalg.inv(Q_uu)
                    
                    # calculate K and d
                    K[t] = -Q_uu_inv @ Q_ux
                    d[t] = -Q_uu_inv @ Q_u
                    
                    # update cost function
                    V_x[t] += K[t].T @ Q_uu @ d[t]
                    V_xx[t] += K[t].T @ Q_uu @ K[t]
                    
                # update control vector
                for t in range(path_len - 1):
                    U[t] = U_init[t] + K[t] @ (X[t] - x_ref_traj[t]) + d[t]
                    if U[t, 1] > 1:
                        U[t, 1] = 1
                    if U[t, 1] < 0.05:
                        U[t, 1] = 0.05
                    if abs(U[t, 0]) > 2:
                        U[t, 0] = 2 * np.sign(U[t, 0])
                    
            
            self.steering_angle, self.speed = U[0]
            self.steering_angle = -self.steering_angle
        
        return np.array([self.steering_angle, self.speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError
        
    def optimal_control(self, pose):
        #### YOUR CODE HERE ####
       
        look_ahead_parameter = 6
        speed_parameter = 10
        look_ahead = np.ceil(self.speed*speed_parameter)
        
        K_p = np.array([[0, 0.3], [0.5, 0]])
        K_i = np.array([[0, 0.005], [0.1, 0.01]])
        K_d = np.array([[0, 1], [0, 0]])
        
        ref_pos = self.ref_traj[(self.waypoint_index + int(look_ahead) - 1) % len(self.ref_traj)]
        dist_to_ref_point = np.linalg.norm(ref_pos - pose[:2])
        
        alpha = pose[2] - np.arctan2(ref_pos[1]-pose[1], ref_pos[0]-pose[0])
        
        self.steering_angle = -np.arctan2(2*np.sin(alpha), dist_to_ref_point)
        
        total_error = np.array([self.current_along_track_error, self.current_cross_track_error])
        total_error = total_error.reshape(2, 1)
        total_integral_error = np.array([self.sum_along_track_error, self.sum_cross_track_error])
        total_integral_error = total_integral_error.reshape(2, 1)
        total_derivative_error = np.array([self.deriv_along_track_error, self.deriv_cross_track_error])
        total_derivative_error = total_derivative_error.reshape(2, 1)
        
        u = - (K_p @ total_error + K_i @ total_integral_error + K_d @ total_derivative_error)
        
        self.speed = float(u[1])
        
        return np.array([self.steering_angle, self.speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError
        

class EndLap(Exception):
    # this exception is raised when the car crosses the finish line
    pass


def main(args=None):
    rclpy.init()

    lab1 = Lab1(controller_type=sys.argv[1])

    tick = time.time()
    try:
        rclpy.spin(lab1)
    except NotImplementedError:
        rclpy.logging.get_logger('lab1').info("You havn't implemented this controller yet!")
    except EndLap:
        tock = time.time()
        rclpy.logging.get_logger('lab1').info("Finished lap")
        rclpy.logging.get_logger('lab1').info("Cross Track Error: " + str(lab1.cross_track_accumulated_error))
        rclpy.logging.get_logger('lab1').info("Along Track Error: " + str(lab1.along_track_accumulated_error))
        rclpy.logging.get_logger('lab1').info("Lap Time: " + str(tock - tick))
        print("Cross Track Error: " + str(lab1.cross_track_accumulated_error))
        print("Along Track Error: " + str(lab1.along_track_accumulated_error))
        print("Lap Time: " + str(tock - tick))

    lab1.destroy_node()
    rclpy.shutdown()
    
    
if __name__ == '__main__':
    controller_type = sys.argv[1]
    main(controller_type)
    
    
        