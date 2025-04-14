################################################
# Descriptions
################################################

'''
node subscribes to camera data from rgb_camera topic
node computes the bounding box via red colour mask
node computes the distances and sends vision_pose and setpoint_position 
accordingly to move the amount required
'''


################################################
# Imports and Setup
################################################

# ros imports
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

# ros imports for realsense and mavros
from geometry_msgs.msg import PoseArray, PoseStamped, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

# reliability imports
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)

# image related
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
import os

bridge = CvBridge()

################################################
# Class Nodes
################################################


class DroneControlNode(Node):

    def __init__(self, test_type):
        super().__init__('drone_control_node') 

        ###############
        # SERVICE CALLS

        self.srv_launch = self.create_service(Trigger, 'rob498_drone_1/comm/launch', self.callback_launch)
        self.srv_test = self.create_service(Trigger, 'rob498_drone_1/comm/test', self.callback_test)
        self.srv_land = self.create_service(Trigger, 'rob498_drone_1/comm/land', self.callback_land)
        self.srv_abort = self.create_service(Trigger, 'rob498_drone_1/comm/abort', self.callback_abort)
        print('services created')

        ###########################
        # USER CONTROLLED VARIABLES

        # tuning parameters
        self.desired_flight_height = 2.3        # height we fly at
        self.max_searching_height = 2.5         # safety on top height
        self.square_size = 3.0                  # safety on the side movement
        self.init_x, self.init_y = 2.0, 1.8     # where we start the drone
        
        # movement parameters 
        self.frame_pixel_tol = 5                # how to center to only hover
        self.Kp = 0.020 #0.0141                 # proportional gain
        self.Kd = 0.002 #0.001                  # derivative gain

        # colour filter settings
        self.target_rgb = (200, 29, 32)         # target color to track, in RGB format (ex: red)
        self.hue_tol = 5                        # tolerance range for hue in HSV space (large = less strict)
        self.sat_tol = 100                      # tolerance for saturation (large = more dull and bright)
        self.val_tol = 100                      # tolerance for value/brightness (large = more lighting conds)
        self.circularity_tol = 0.8              # how circle we want to be
        self.score_tol = 7                      # how much we want to be above the threshold

        ###########################
        # OTHER SETUP (DON'T TOUCH)

        # init the colour filter
        self.set_target_color()

        # safety net on the ball
        self.bounds = {"x_min": -1*self.square_size, "x_max": self.square_size, "y_min": -1*self.square_size, "y_max": self.square_size, "z_min": 0.0, "z_max": self.max_searching_height}

        # for vision_pose to know where it is
        self.position = Point()
        self.orientation = Quaternion()
        self.timestamp = None
        self.frame_id = "map"

        # for setpoint_vision to know where to go
        self.set_position = Point()
        self.set_orientation = Quaternion()
        self.set_orientation.w = -1.0

        # booleans for enabling testing
        self.testing = False
        self.t1 = time.time()
        self.t2 = time.time()

        # camera parameters
        self.REAL_DIAMETER_MM = 42.67       # Standard golf ball diameter in mm
        self.FOCAL_LENGTH_MM = 26           # iPhone 14 Plus main camera focal length in mm
        self.SENSOR_WIDTH_MM = 4.93         # Approximate sensor size: 5.095 mm (H) × 4.930 mm (W)
        self.DOWN_SAMPLE_FACTOR = 4         # Downsample factor used in calculation

        # frame parameters (updated in first frame)
        self.frame_width, self.frame_height = None, None
        self.camera_frame_center = None
        self.FOCAL_LENGTH_PIXELS = None

        # derivative parameters
        self.prev_p_error_x = 0
        self.prev_p_error_y = 0
        self.prev_time = self.get_clock().now()
        
        ############################
        # SUBSCRIBER/PUBLISHER SETUP

        # ROS subscriber to RGB camera messages
        self.camera_subscriber = self.create_subscription(Image, '/camera/image_raw', self.frame_input_callback, 1)
        self.get_logger().info('Subscribed to Camera Input!')
        self.br = CvBridge()

        self.image_publisher = self.create_publisher(Image, '/camera/segmented', 1)
        self.get_logger().info('Publishing to Processed Camera Output!')

        # subscriber to RealSense or Vicon pose data
        if test_type == "realsense":
            # Subscriber to RealSense pose data
            self.realsense_subscriber = self.create_subscription(Odometry, '/camera/pose/sample', self.realsense_callback, qos_profile)
            self.get_logger().info('Subscribing to RealSense!')
        else: 
            # Subscriber to Vicon pose data
            self.vicon_subscriber = self.create_subscription(PoseStamped, '/vicon/ROB498_Drone/ROB498_Drone', self.vicon_callback, 1)
            self.get_logger().info('Subscribing to Vicon!')
        
        # publisher for VisionPose topic
        self.vision_pose_publisher = self.create_publisher(PoseStamped, '/mavros/vision_pose/pose', 1)
        self.get_logger().info('Publishing to VisionPose')

        # publisher for SetPoint topic
        self.setpoint_publisher = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', qos_profile)
        self.get_logger().info('Publishing to SetPoint')

        # statement to end the inits
        self.get_logger().info('Nodes All Setup and Started!')

    ################################################
    # SERVICE CALLS 
    ################################################

    def callback_launch(self, request, response):
        print('Launch Requested. Drone takes off to find the golf ball and hover overtop.')
        self.launching_procedure()
        return response

    def callback_test(self, request, response):
        print('Test Requested. Drone is read_error_y to follow whever the ball may go.')
        self.testing_procedure()
        return response
        
    def callback_land(self, request, response):
        print('Land Requested. Drone will return to starting position where the humans are.')
        self.landing_procedure()
        return response

    def callback_abort(self, request, response):
        print('Abort Requested. Drone will land immediately due to safety considerations.')
        self.abort_procedure()
        return response

    ################################################
    # SERVICE FUNCTIONS
    ################################################

    def launching_procedure(self):
        # start by taking off and flying higher to search for the ball
        # continuously search until a valid center point is found
        # once the ball is detected, lower the drone to the desired height
        # center the drone over the ball
        # capture the current position for landing
        self.set_pose_initial()
        self.set_position.z = self.desired_flight_height
        return

    def testing_procedure(self):
        # set the drone to continuously hover and track the ball
        self.testing = True
        return

    def landing_procedure(self):
        # drone will land at the captured position (back where the people are)
        # also at a lower height
        self.testing = False
        self.set_position.z = 0.1
        return

    def abort_procedure(self):
        # safety land will just immediately lower the drone
        self.testing = False
        self.set_position.z = 0.0
        response.success = True
        response.message = "Success"

    ################################################
    # CALLBACKS
    ################################################

    def realsense_callback(self, msg):
        # get the info
        self.position = msg.pose.pose.position
        self.orientation = msg.pose.pose.orientation
        self.timestamp = self.get_clock().now().to_msg()
        # frame conversion
        self.orientation.x *= -1
        self.orientation.y *= -1
        self.orientation.z *= -1
        self.orientation.w *= -1
        # WRITE BOTH IMMEDIATELY
        self.send_vision_pose()
        self.send_setpoint()

    def vicon_callback(self, msg):
        # get the info
        self.position = msg.pose.position
        self.orientation = msg.pose.orientation
        self.timestamp = self.get_clock().now().to_msg()
        # frame conversion
        self.orientation.x *= -1
        self.orientation.y *= -1
        self.orientation.z *= -1
        self.orientation.w *= -1
        # WRITE BOTH IMMEDIATELY
        self.send_vision_pose()
        self.send_setpoint()

    def send_vision_pose(self):
        # Create a new PoseStamped message to publish to vision_pose topic
        vision_pose_msg = PoseStamped()
        vision_pose_msg.header.stamp = self.timestamp
        vision_pose_msg.header.frame_id = self.frame_id
        vision_pose_msg.pose.position = self.position
        vision_pose_msg.pose.orientation = self.orientation
        # Publish the message to the /mavros/vision_pose/pose topic
        self.vision_pose_publisher.publish(vision_pose_msg)

    def clamp_position(self, position):
        # Apply safety bounds to the setpoints so the drone never tries to go outside
        position.x = max(self.bounds["x_min"], min(position.x, self.bounds["x_max"]))
        position.y = max(self.bounds["y_min"], min(position.y, self.bounds["y_max"]))
        position.z = max(self.bounds["z_min"], min(position.z, self.bounds["z_max"]))
        return position

    def send_setpoint(self):
        # Create a new PoseStamped message to publish to setpoint topic
        current_position = self.clamp_position(self.set_position)
        setpoint_msg = PoseStamped()
        setpoint_msg.header.stamp = self.timestamp
        setpoint_msg.header.frame_id = self.frame_id
        setpoint_msg.pose.position = current_position
        setpoint_msg.pose.orientation = self.set_orientation
        # Publish the message to the /mavros/setpoint_position/local topic
        self.setpoint_publisher.publish(setpoint_msg)

    def frame_input_callback(self, msg):
        # convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(msg)
        # run the full processing on the frame to change setpoint
        self.full_image_processing(current_frame)
        return

    ################################################
    # IMAGE PROCESSING
    ################################################

    def full_image_processing(self, frame):
        self.t1 = time.time()
        # the first time, we set up parameters
        if self.FOCAL_LENGTH_PIXELS is None: self.first_time_setup_image_parameters(frame)
        # take the frame and find the object center
        center = self.find_object_center(frame)
        # if the center exists, we assign to current ball position
        if center:
            self.curr_center = center
            # draw the center on the frame
            cv2.circle(frame, self.curr_center, 1, (0, 255, 0), -1)
            # calculate the offset from the frame center
            offset_x_pixels, offset_y_pixels = self.mini_calculate_golf_ball_metrics()
            # then based on how far off we are, instruct the drone's setpoint to move that much
            self.move_drone(offset_x_pixels, offset_y_pixels)
        # always publish the images regadless if a frame was drawn in or not
        self.image_publisher.publish(bridge.cv2_to_imgmsg(frame))
        return

    def find_object_center(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
        #######
        # remove green
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        # remove blue
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # invert green and blue masks
        not_green = cv2.bitwise_not(green_mask)
        not_blue = cv2.bitwise_not(blue_mask)
        # combine masks: keep red, remove green & blue
        combined_mask = cv2.bitwise_and(mask, not_green)
        combined_mask = cv2.bitwise_and(combined_mask, not_blue)
        #######
        mask = cv2.GaussianBlur(combined_mask, (5, 5), 2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour, best_center, best_score, valid_contours = None, None, 0.0, []
        for cnt in contours:
            valid_contours.append(cnt)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0 or area < 20: continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > self.circularity_tol:
                score = circularity * 2*np.log(area)
                if score > best_score and score > self.score_tol:
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        best_contour = cnt
                        best_center = (cx, cy)
                        best_score = score
                        # print(f"Best contour score: {best_score:.2f}")
        return best_center


    def mini_calculate_golf_ball_metrics(self):
        # using the frame center and current ball center, find offset
        offset_x_pixels = self.curr_center[0] - self.camera_frame_center[0]
        offset_y_pixels = self.curr_center[1] - self.camera_frame_center[1]
        return offset_x_pixels, offset_y_pixels


    def move_drone(self, p_error_x, p_error_y):
        # calculate the vector length
        vector_length = self.calculate_pixel_difference(p_error_x, p_error_y)
        print("moving drone triggered", vector_length)
        # if the length is close enough, no change to setpoint, we don't move
        if vector_length <= self.frame_pixel_tol: 
            print("*** HOVERING ***")
            if self.testing:
                self.set_position.x = self.position.x
                self.set_position.y = self.position.y
                self.set_position.z = self.desired_flight_height
            return
        # if we made it past here, then we want to move
        print("*** MOVE ***")
        # calculate derivative component
        curr_time = self.get_clock().now()
        dt = (curr_time - self.prev_time).nanoseconds / 1e9  # convert ns to seconds
        if dt == 0: dt = 1e-6  # prevent division by zero
        d_error_x = (p_error_x - self.prev_p_error_x) / dt
        d_error_y = (p_error_y - self.prev_p_error_y) / dt
        # PD control signal
        move_x = self.Kp * p_error_x + self.Kd * d_error_x
        move_y = self.Kp * p_error_y + self.Kd * d_error_y
        # update the drone's position with the scaled values
        if self.testing:
            self.set_position.x = self.position.x - move_y
            self.set_position.y = self.position.y - move_x
            self.set_position.z = self.desired_flight_height
        # save current state for next iteration
        self.prev_p_error_x = p_error_x
        self.prev_p_error_y = p_error_y
        self.prev_time = curr_time
        self.t2 = time.time()

        print(self.t2 - self.t1)


    ################################################
    # IMAGE PROCESSING HELPERS
    ################################################

    def rgb_to_hsv(self, rgb):
        # convert RGB to HSV
        rgb_array = np.uint8([[list(rgb)]])
        hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        return tuple(hsv_array[0][0]) 

    def set_target_color(self):
        # set the colour and give bounds for tolerance
        hsv_color = self.rgb_to_hsv()
        h, s, v = hsv_color
        self.lower_bound = np.array([
            max(h - self.hue_tol, 0),
            max(s - self.sat_tol, 0),
            max(v - self.val_tol, 0)])
        self.upper_bound = np.array([
            min(h + self.hue_tol, 179),
            min(s + self.sat_tol, 255),
            min(v + self.val_tol, 255)])

    def calculate_pixel_difference(self, x, y):
        # calculate vector lengths
        vector_length = (x ** 2 + y ** 2) ** 0.5
        return vector_length

    def first_time_setup_image_parameters(self, frame):
        self.frame_height, self.frame_width, _ = frame.shape
        self.camera_frame_center = (self.frame_width / 2, self.frame_height / 2)
        self.FOCAL_LENGTH_PIXELS = ((self.FOCAL_LENGTH_MM / self.SENSOR_WIDTH_MM) * self.frame_width) / self.DOWN_SAMPLE_FACTOR
        return

    def set_pose_initial(self):
        # Put the current position into maintained position
        self.set_position.x = self.init_x
        self.set_position.y = self.init_y
        self.set_position.z = 0.0
        self.set_orientation.x = 0.0
        self.set_orientation.y = 0.0
        self.set_orientation.z = 0.0
        self.set_orientation.w = -1.0


################################################
# MAIN EXECUTION
################################################

def main(args=None):
    rclpy.init(args=args)
    test_type = "vicon"
    node = DroneControlNode(test_type)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('SHUTTING DOWN NODE.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()


################################################
# END
################################################





##################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################




# ################################################
# # Descriptions
# ################################################

# '''
# node subscribes to camera data from rgb_camera topic
# node computes the bounding box via yolo
# node computes the distances and sends vision_pose and setpoint_position 
# accordingly to move the amount required

# ** there may be issues with synch between realsense pose and rgb vision
# ** should be robust enough with some lag
# ** better at higher altitudes

# 1. maybe some timing thing (like frame id or that clock thing) needs to be leveraged to sync the realsense data and the segmentation data?
# 2. can use the confidence thing maybe if we change to pid to know how much to weigh
# 3. technically i don't think the frame needs to be passed over, but might be nice to have all the data here
# 4. this assignment thing on setpoint the directions might not be right (can also incorporate distance somehow, which doesn't have a around 0 operating point)

# not done the actual landing testing and launching procedures yet but they are mapped out
# '''


# ################################################
# # Imports and Setup
# ################################################

# import os
# os.environ["QT_QPA_PLATFORM"] = "offscreen"
# os.environ["DISPLAY"] = ""

# # ros imports
# import rclpy
# from rclpy.node import Node
# from std_srvs.srv import Trigger

# # ros imports for realsense and mavros
# from geometry_msgs.msg import PoseArray, PoseStamped, Point, Quaternion
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import Image

# # reliability imports
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy
# qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)

# # image related
# import cv2
# from cv_bridge import CvBridge
# import torch
# import numpy as np
# from ultralytics import YOLO
# import matplotlib.pyplot as plt
# import time

# # other imports
# import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# PARSIGHT_PATH = "./src/ParSight/ParSight"
# model = YOLO("/home/jetson/flyrs_ws/src/ParSight/ParSight/models/best_128_new.engine")
# bridge = CvBridge()
# print(torch.cuda.is_available())
# print("Running on:", model.device)


# ################################################
# # Class Nodes
# ################################################


# class SegDroneControlNode(Node):

#     def __init__(self, test_type):
#         super().__init__('seg_drone_control_node') 

#         ###############
#         # SERVICE CALLS

#         self.srv_launch = self.create_service(Trigger, 'rob498_drone_1/comm/launch', self.callback_launch)
#         self.srv_test = self.create_service(Trigger, 'rob498_drone_1/comm/test', self.callback_test)
#         self.srv_land = self.create_service(Trigger, 'rob498_drone_1/comm/land', self.callback_land)
#         self.srv_abort = self.create_service(Trigger, 'rob498_drone_1/comm/abort', self.callback_abort)
#         print('services created')

#         #######################
#         # MAVROS VARIABLE SETUP

#         # tuning parameters
#         self.desired_flight_height = 1.8    # height we fly at
#         self.max_searching_height = 2.0     # safety on top height
#         self.square_size = 3.0              # safety on the side movement
#         self.threshold_confidence = 0.2     # yolo segmentation
#         self.move_amount = 0.30             # how much to assign movement per frame
#         self.frame_pixel_tol = 5            # how to center to only hover
#         self.jump_pixel_threshold = 20      # for dynamic motion

#         # safety net on the ball
#         self.bounds = {"x_min": -1*self.square_size, "x_max": self.square_size, "y_min": -1*self.square_size, "y_max": self.square_size, "z_min": 0.0, "z_max": self.max_searching_height}

#         # for vision_pose to know where it is
#         self.position = Point()
#         self.orientation = Quaternion()
#         self.timestamp = None
#         self.frame_id = "map"

#         # for setpoint_vision to know where to go
#         self.set_position = Point()
#         self.set_orientation = Quaternion()
#         self.set_orientation.w = -1.0

#         ######################
#         # IMAGE VARIABLE SETUP

#         # init class attributes to store msg: image, bbox, confidence, and validity
#         self.image_data = None              # from segmentation (raw)      
#         self.bbox_data = None               # from segmentation (raw)      
#         self.confidence_data = None         # from segmentation (raw)      
#         self.curr_bbox = None               # used for tracking when valid
#         self.last_bbox = None               # dynamic matching 
#         self.bbox_counting = 0              # 
#         self.bbox_counting_max = 3

#         # booleans for enabling testing
#         self.testing = False
#         self.valid_bbox = False
#         self.yes_bbox_got = False
#         self.cut_looping = False

#         # camera parameters
#         self.REAL_DIAMETER_MM = 42.67       # Standard golf ball diameter in mm
#         self.FOCAL_LENGTH_MM = 26           # iPhone 14 Plus main camera focal length in mm
#         self.SENSOR_WIDTH_MM = 4.93         # Approximate sensor size: 5.095 mm (H) × 4.930 mm (W)
#         self.DOWN_SAMPLE_FACTOR = 4         # Downsample factor used in YOLO model

#         # frame parameters (updated in first frame)
#         self.frame_width, self.frame_height = None, None
#         self.camera_frame_center = None
#         self.FOCAL_LENGTH_PIXELS = None
        
#         ############################
#         # SUBSCRIBER/PUBLISHER SETUP

#         # ROS subscriber to RGB camera messages
#         self.camera_subscriber = self.create_subscription(Image, '/camera/image_raw', self.frame_input_callback, 10)
#         self.get_logger().info('Subscribed to Camera Input!')
#         self.br = CvBridge()

#         self.image_publisher = self.create_publisher(Image, '/camera/segmented', 1)
#         self.get_logger().info('Publishing to Processed Camera Output!')

#         # subscriber to RealSense or Vicon pose data
#         if test_type == "realsense":
#             # Subscriber to RealSense pose data
#             self.realsense_subscriber = self.create_subscription(Odometry, '/camera/pose/sample', self.realsense_callback, qos_profile)
#             self.get_logger().info('Subscribing to RealSense!')
#         else: 
#             # Subscriber to Vicon pose data
#             self.vicon_subscriber = self.create_subscription(PoseStamped, '/vicon/ROB498_Drone/ROB498_Drone', self.vicon_callback, 1)
#             self.get_logger().info('Subscribing to Vicon!')
        
#         # publisher for VisionPose topic
#         self.vision_pose_publisher = self.create_publisher(PoseStamped, '/mavros/vision_pose/pose', 1)
#         self.get_logger().info('Publishing to VisionPose')

#         # publisher for SetPoint topic
#         self.setpoint_publisher = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', qos_profile)
#         self.get_logger().info('Publishing to SetPoint')

#         # statement to end the inits
#         self.get_logger().info('Nodes All Setup and Started!')

#     ################################################
#     # SERVICE CALLS 
#     ################################################

#     def callback_launch(self, request, response):
#         print('Launch Requested. Drone takes off to find the golf ball and hover overtop.')
#         self.launching_procedure()
#         return response

#     def callback_test(self, request, response):
#         print('Test Requested. Drone is ready to follow whever the ball may go.')
#         self.testing_procedure()
#         return response
        
#     def callback_land(self, request, response):
#         print('Land Requested. Drone will return to starting position where the humans are.')
#         self.landing_procedure()
#         return response

#     def callback_abort(self, request, response):
#         print('Abort Requested. Drone will land immediately due to safety considerations.')
#         self.abort_procedure()
#         return response

#     ################################################
#     # SERVICE FUNCTIONS
#     ################################################

#     def launching_procedure(self):
#         # start by taking off and flying higher to search for the ball
#         # continuously search until a valid segmentation is found
#         # once the ball is detected, lower the drone to the desired height
#         # center the drone over the ball
#         # capture the current position for landing
#         self.set_pose_initial()
#         self.set_position.z = self.desired_flight_height
#         return

#     def testing_procedure(self):
#         # set the drone to continuously hover and track the ball
#         self.testing = True
#         return

#     def landing_procedure(self):
#         # drone will land at the captured position (back where the people are)
#         # also at a lower height
#         self.testing = False
#         self.set_position.z = 0.1
#         return

#     def abort_procedure(self):
#         # safety land will just immediately lower the drone
#         self.testing = False
#         self.set_position.z = 0.0
#         response.success = True
#         response.message = "Success"

#     ################################################
#     # CALLBACKS
#     ################################################

#     def realsense_callback(self, msg):
#         # get the info
#         self.position = msg.pose.pose.position
#         self.orientation = msg.pose.pose.orientation
#         self.timestamp = self.get_clock().now().to_msg()
#         # frame conversion
#         self.orientation.x *= -1
#         self.orientation.y *= -1
#         self.orientation.z *= -1
#         self.orientation.w *= -1
#         # WRITE BOTH IMMEDIATELY
#         self.send_vision_pose()
#         self.send_setpoint()

#     def vicon_callback(self, msg):
#         # get the info
#         self.position = msg.pose.position
#         self.orientation = msg.pose.orientation
#         self.timestamp = self.get_clock().now().to_msg()
#         # frame conversion
#         self.orientation.x *= -1
#         self.orientation.y *= -1
#         self.orientation.z *= -1
#         self.orientation.w *= -1
#         # WRITE BOTH IMMEDIATELY
#         self.send_vision_pose()
#         self.send_setpoint()

#     def send_vision_pose(self):
#         # Create a new PoseStamped message to publish to vision_pose topic
#         vision_pose_msg = PoseStamped()
#         vision_pose_msg.header.stamp = self.timestamp
#         vision_pose_msg.header.frame_id = self.frame_id
#         vision_pose_msg.pose.position = self.position
#         vision_pose_msg.pose.orientation = self.orientation
#         # Publish the message to the /mavros/vision_pose/pose topic
#         self.vision_pose_publisher.publish(vision_pose_msg)

#     def clamp_position(self, position):
#         # Apply safety bounds to the setpoints so the drone never tries to go outside
#         position.x = max(self.bounds["x_min"], min(position.x, self.bounds["x_max"]))
#         position.y = max(self.bounds["y_min"], min(position.y, self.bounds["y_max"]))
#         position.z = max(self.bounds["z_min"], min(position.z, self.bounds["z_max"]))
#         return position

#     def send_setpoint(self):
#         # Create a new PoseStamped message to publish to setpoint topic
#         current_position = self.clamp_position(self.set_position)
#         setpoint_msg = PoseStamped()
#         setpoint_msg.header.stamp = self.timestamp
#         setpoint_msg.header.frame_id = self.frame_id
#         setpoint_msg.pose.position = current_position
#         setpoint_msg.pose.orientation = self.set_orientation
#         # Publish the message to the /mavros/setpoint_position/local topic
#         self.setpoint_publisher.publish(setpoint_msg)

#     def frame_input_callback(self, msg):
#         # convert ROS Image message to OpenCV image
#         current_frame = self.br.imgmsg_to_cv2(msg)
#         # run the yolo segmentation
#         self.image_data, self.bbox_data, self.confidence_data = self.run_yolo_segmentation(current_frame)
#         if self.bbox_data != [-1, -1, -1, -1]: self.valid_bbox = True; self.yes_bbox_got = True
#         else: self.valid_bbox = False
#         # these are just print statements
#         print(f"RECEIVED: || BBox: {self.bbox_data} || Conf: {self.confidence_data}")
#         ########################################################
#         # then we go into any image processing
#         self.full_image_processing()
#         return

#     ################################################
#     # IMAGE PROCESSING
#     ################################################

#     def run_yolo_segmentation(self, frame):
#         # apply a confidence threshold
#         results = model(frame, imgsz=128, conf=0.0, verbose=True)
#         # initialize variables for the highest confidence detection
#         best_conf, best_bbox = 0, None
#         # cycle through all found bboxes
#         for result in results:
#             for det in result.boxes.data:
#                 # extract the bounding box
#                 x_min, y_min, x_max, y_max, conf, cls = det.tolist()
#                 conf = float(conf)
#                 # filter low confidence out
#                 if conf < self.threshold_confidence: continue
#                 # check if this is the highest confidence so far
#                 if conf > best_conf:
#                     best_conf = conf
#                     best_bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
#         # drawing the bounding box on the frame
#         if best_bbox is not None:
#             # something was recieved, we now want to make sure it's possible
#             if self.last_bbox is not None:
#                 # we maybe segmented a false positive, assume instead nothing was recieved
#                 x_cur, y_cur = self.find_center_point(best_bbox)
#                 x_last, y_last = self.find_center_point(self.last_bbox)
#                 x_off, y_off = x_cur-x_last, y_cur-y_last
#                 if self.calculate_pixel_difference(x_off, y_off) > self.jump_pixel_threshold: 
#                     best_bbox, conf = [-1, -1, -1, -1], -2
#                     self.bbox_counting += 1
#                     print("----------------------------------------JUMPED")
#                 else:
#                     # draw the bounding box on the frame
#                     cv2.rectangle(frame, (best_bbox[0], best_bbox[1]), (best_bbox[2], best_bbox[3]), (0, 255, 0), 1)
#                     label = f'Conf: {best_conf:.2f}'
#                     cv2.putText(frame, label, (best_bbox[0]-5, best_bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#             else:
#                 # draw the bounding box on the frame
#                 cv2.rectangle(frame, (best_bbox[0], best_bbox[1]), (best_bbox[2], best_bbox[3]), (0, 255, 0), 1)
#                 label = f'Conf: {best_conf:.2f}'
#                 cv2.putText(frame, label, (best_bbox[0]-5, best_bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#         # if no valid bounding box found
#         else: 
#             best_bbox, conf = [-1, -1, -1, -1], -1
#         # always publish the images regadless if a frame was drawn in or not
#         self.image_publisher.publish(bridge.cv2_to_imgmsg(frame))
#         # reset last bbox if for too long, the jumping is being bad
#         if self.bbox_counting >= self.bbox_counting_max:
#             self.bbox_counting = 0
#             self.last_bbox = None
#         # send raw results
#         return frame, best_bbox, conf

#     def full_image_processing(self):
#         # the first time, we set up parameters
#         if self.FOCAL_LENGTH_PIXELS is None: self.first_time_setup_image_parameters()
#         # safety check in case the new bbox is bad, use last
#         if self.valid_bbox: self.curr_bbox = self.bbox_data
#         # then everytime we get the distances
#         if self.yes_bbox_got:
#             offset_x_pixels, offset_y_pixels = self.mini_calculate_golf_ball_metrics()
#             # then based on how far off we are, instruct the drone's setpoint to move that much
#             self.move_drone(offset_x_pixels, offset_y_pixels)
#             self.last_bbox = self.curr_bbox
    
#     def mini_calculate_golf_ball_metrics(self):
#         # calculate ball and image centers (in pixel coordinates)
#         ball_center_x, ball_center_y = self.find_center_point(self.curr_bbox)
#         image_center_x, image_center_y = self.camera_frame_center
#         offset_x_pixels, offset_y_pixels = ball_center_x - image_center_x, ball_center_y - image_center_y
#         return offset_x_pixels, offset_y_pixels
 
#     def move_drone(self, offset_x_pixels, offset_y_pixels):
#         # calculate the vector length
#         vector_length = self.calculate_pixel_difference(offset_x_pixels, offset_y_pixels)
#         print("moving drone triggered", vector_length)
#         # if the length is close enough, no change to setpoint, we don't move
#         if vector_length <= self.frame_pixel_tol: print("*** HOVERING ***"); return
#         # if we made it past here, then we want to move
#         print("*** MOVE ***")
#         scaled_x = (offset_x_pixels / vector_length) * self.move_amount
#         scaled_y = (offset_y_pixels / vector_length) * self.move_amount
#         # update the drone's position with the scaled values
#         if self.testing:
#             self.set_position.x = self.position.x - scaled_y
#             self.set_position.y = self.position.y - scaled_x
#             self.set_position.z = self.desired_flight_height
#         return


#     ################################################
#     # IMAGE PROCESSING HELPERS
#     ################################################

#     def find_center_point(self, bbox):
#         x1, y1, x2, y2 = bbox
#         bbox_width, bbox_height = x2 - x1, y2 - y1
#         ball_center_x, ball_center_y = x1 + bbox_width / 2, y1 + bbox_height / 2
#         return ball_center_x, ball_center_y

#     def calculate_pixel_difference(self, x, y):
#         # calculate vector lengths
#         vector_length = (x ** 2 + y ** 2) ** 0.5
#         return vector_length

#     def imgmsg_to_numpy(self, ros_image):
#         # helper to convert ROS image to numpy array
#         bridge = CvBridge()
#         return bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')

#     def first_time_setup_image_parameters(self):
#         self.frame_height, self.frame_width, _ = self.image_data.shape
#         self.camera_frame_center = (self.frame_width / 2, self.frame_height / 2)
#         self.FOCAL_LENGTH_PIXELS = ((self.FOCAL_LENGTH_MM / self.SENSOR_WIDTH_MM) * self.frame_width) / self.DOWN_SAMPLE_FACTOR
#         return

#     def set_pose_initial(self):
#         # Put the current position into maintained position
#         self.set_position.x = 0.0
#         self.set_position.y = 0.0
#         self.set_position.z = 0.0
#         self.set_orientation.x = 0.0
#         self.set_orientation.y = 0.0
#         self.set_orientation.z = 0.0
#         self.set_orientation.w = -1.0

#     ################################################
#     # IMAGE PROCESSING HELPERS
#     ################################################

#     # def pixels_to_meters(self, pixel_offset, distance_m):
#     #     # Compute displacement in mm, then convert to meters
#     #     return (distance_m / self.FOCAL_LENGTH_PIXELS) * pixel_offset

#     # def meters_to_pixels(self, offset_m, distance_m):
#     #     # Convert offset to mm, then compute pixel displacement
#     #     return (offset_m * self.FOCAL_LENGTH_PIXELS) / distance_m

#     # def calculate_golf_ball_metrics(self):
#     #     # unpack all the values from the bounding box and calculate the diameter
#     #     x1, y1, x2, y2 = self.curr_bbox
#     #     bbox_width, bbox_height = x2 - x1, y2 - y1
#     #     # if the ball is cut off on the edges, choose the larger dimension
#     #     if x1 <= 0 or y1 <= 0 or x2 >= self.frame_width or y2 >= self.frame_height: diameter_pixels = max(bbox_width, bbox_height)
#     #     else: diameter_pixels = (bbox_width + bbox_height) / 2
#     #     # compute the golf ball's center
#     #     ball_center_x, ball_center_y = x1 + bbox_width / 2, y1 + bbox_height / 2
#     #     image_center_x, image_center_y = self.camera_frame_center
#     #     # compute the distance to the ball
#     #     distance_mm = (self.REAL_DIAMETER_MM * self.FOCAL_LENGTH_PIXELS) / diameter_pixels
#     #     distance_m = distance_mm / 1000 
#     #     # calculate ball and image centers (in pixel coordinates)
#     #     offset_x_pixels, offset_y_pixels = ball_center_x - image_center_x, ball_center_y - image_center_y
#     #     offset_x_m = self.pixels_to_meters(offset_x_pixels, distance_m)
#     #     offset_y_m = self.pixels_to_meters(offset_y_pixels, distance_m)
#     #     return distance_m, offset_x_m, offset_y_m, offset_x_pixels, offset_y_pixels

# ################################################
# # MAIN EXECUTION
# ################################################

# def main(args=None):
#     rclpy.init(args=args)
#     test_type = "vicon"
#     node = SegDroneControlNode(test_type)
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('SHUTTING DOWN NODE.')
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == "__main__":
#     main()


# ################################################
# # END
# ################################################