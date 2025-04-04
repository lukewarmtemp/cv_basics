################################################
# Descriptions
################################################

'''
node subscribes to camera data from rgb_camera topic
node computes the bounding box via yolo
node computes the distances and sends vision_pose and setpoint_position 
accordingly to move the amount required

** there may be issues with synch between realsense pose and rgb vision
** should be robust enough with some lag
** better at higher altitudes

1. maybe some timing thing (like frame id or that clock thing) needs to be leveraged to sync the realsense data and the segmentation data?
2. can use the confidence thing maybe if we change to pid to know how much to weigh
3. technically i don't think the frame needs to be passed over, but might be nice to have all the data here
4. this assignment thing on setpoint the directions might not be right (can also incorporate distance somehow, which doesn't have a around 0 operating point)

not done the actual landing testing and launching procedures yet but they are mapped out
'''


################################################
# Imports and Setup
################################################

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""

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
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time

# other imports
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PARSIGHT_PATH = "./src/ParSight/ParSight"
model = YOLO("/home/jetson/flyrs_ws/src/ParSight/ParSight/models/best_128.engine")
bridge = CvBridge()
print(torch.cuda.is_available())
print("Running on:", model.device)


################################################
# Class Nodes
################################################


class SegDroneControlNode(Node):

    def __init__(self, test_type, drone_pose_tool):
        super().__init__('seg_drone_control_node') 

        ###############
        # SERVICE CALLS

        self.srv_launch = self.create_service(Trigger, 'rob498_drone_1/comm/launch', self.callback_launch)
        self.srv_test = self.create_service(Trigger, 'rob498_drone_1/comm/test', self.callback_test)
        self.srv_land = self.create_service(Trigger, 'rob498_drone_1/comm/land', self.callback_land)
        self.srv_abort = self.create_service(Trigger, 'rob498_drone_1/comm/abort', self.callback_abort)
        print('services created')

        #######################
        # MAVROS VARIABLE SETUP

        # generally how high above the ball we fly
        self.desired_flight_height = 1.0
        self.max_searching_height = 1.5
        self.square_size = 1.0
        self.bounds = {"x_min": -1*self.square_size, "x_max": self.square_size, "y_min": -1*self.square_size, "y_max": self.square_size, "z_min": 0.0, "z_max": 1.0}

        # for vision_pose to know where it is
        self.position = Point()
        self.orientation = Quaternion()
        self.timestamp = None
        self.frame_id = "map"

        # for setpoint_vision to know where to go
        self.set_position = Point()
        self.set_orientation = Quaternion()
        self.set_orientation.w = -1.0

        ######################
        # IMAGE VARIABLE SETUP

        # init class attributes to store msg: image, bbox, confidence, and validity
        self.drone_pose_tool = drone_pose_tool
        self.image_data = None
        self.bbox_data = None
        self.confidence_data = None
        self.valid_bbox = False
        self.yes_bbox_got = False
        self.cut_looping = False
        self.curr_bbox = None

        # camera parameters
        self.REAL_DIAMETER_MM = 42.67  # Standard golf ball diameter in mm
        self.FOCAL_LENGTH_MM = 26      # iPhone 14 Plus main camera focal length in mm
        self.SENSOR_WIDTH_MM = 4.93     # Approximate sensor size: 5.095 mm (H) × 4.930 mm (W)
        self.DOWN_SAMPLE_FACTOR = 4    # Downsample factor used in YOLO model

        # frame parameters (updated in first frame)
        self.frame_width, self.frame_height = None, None
        self.camera_frame_center = None
        self.FOCAL_LENGTH_PIXELS = None
        
        self.testing = False

        ############################
        # SUBSCRIBER/PUBLISHER SETUP

        # ROS subscriber to RGB camera messages
        self.camera_subscriber = self.create_subscription(Image, '/camera/image_raw', self.frame_input_callback, 10)
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
        print('Test Requested. Drone is ready to follow whever the ball may go.')
        self.testing_procedure()
        return response
        
    def callback_land(self, request, response):
        print('Land Requested. Drone will return to starting position where the humans are.')
        self.landing_procedure()
        return response

    def callback_abort(self, request, response):
        print('Abort Requested. Drone will land immediately due to safety considerations.')
        self.set_position.z = 0.0
        response.success = True
        response.message = "Success"
        return response

    ################################################
    # SERVICE FUNCTIONS
    ################################################

    def launching_procedure(self):
        # start by taking off and flying higher to search for the ball
        # continuously search until a valid segmentation is found
        # once the ball is detected, lower the drone to the desired height
        # center the drone over the ball
        # capture the current position for landing
        # TODO
        self.set_pose_initial()
        self.set_position.z = self.desired_flight_height
        return

    def testing_procedure(self):
        # set the drone to continuously hover and track the ball
        # TODO
        self.testing = True
        return

    def landing_procedure(self):
        # drone will land at the captured position (back where the people are)
        # also at a lower height
        # TODO
        self.testing = False
        self.set_position.z = 0.1
        return

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
        # print(f"Position: x={self.set_position.x}, y={self.set_position.y}, z={self.set_position.z}")
        # print(f"Orientation: x={self.orientation.x}, y={self.orientation.y}, z={self.orientation.z}, w={self.orientation.w}")
        # print(f"Timestamp: {self.timestamp.sec}.{self.timestamp.nanosec}")
        # print(f"Frame ID: {self.frame_id}")
        # Publish the message to the /mavros/setpoint_position/local topic
        self.setpoint_publisher.publish(setpoint_msg)

    def frame_input_callback(self, msg):
        # convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(msg)
        # run the yolo segmentation
        self.image_data, self.bbox_data, self.confidence_data = self.run_yolo_segmentation(current_frame)
        if self.bbox_data != [-1, -1, -1, -1]: self.valid_bbox = True; self.yes_bbox_got = True
        else: self.valid_bbox = False
        # these are just print statements
        print(f"Received image with bbox: {self.bbox_data} and confidence: {self.confidence_data}")
        # print(f"Valid BBox: {self.valid_bbox}")
        ########################################################
        # then we go into any image processing
        self.full_image_processing()
        # cv2.imshow("camera", current_frame)
        # cv2.waitKey(1)
        return

    ################################################
    # IMAGE PROCESSING
    ################################################

    def run_yolo_segmentation(self, frame):
        # start_time = time.time()

        # apply a confidence threshold
        results = model(frame, imgsz=128, conf=0.4, verbose=True)
        # initialize variables for the highest confidence detection
        best_conf, best_bbox = 0, None
        # cycle through all found bboxes
        for result in results:
            for det in result.boxes.data:
                # extract the bounding box
                x_min, y_min, x_max, y_max, conf, cls = det.tolist()
                conf = float(conf)
                # filter low confidence out
                if conf < 0.2: continue
                # check if this is the highest confidence so far
                if conf > best_conf:
                    best_conf = conf
                    best_bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        # drawing the bounding box on the frame
        if best_bbox is not None:
            # draw the bounding box on the frame
            cv2.rectangle(frame, (best_bbox[0], best_bbox[1]), (best_bbox[2], best_bbox[3]), (0, 255, 0), 2)
            label = f'Conf: {best_conf:.2f}'
            cv2.putText(frame, label, (best_bbox[0], best_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #cv2.imshow("BB Frame",frame)
            #cv2.waitKey(10)
        self.image_publisher.publish(bridge.cv2_to_imgmsg(frame))
        # if no valid bounding box found, set it to [-1, -1, -1, -1]
        if best_bbox is None:
            best_bbox = [-1, -1, -1, -1]
            conf = -1
        
        # end_time = time.time()
        # total_time = end_time - start_time
        # print(total_time, best_bbox)

        return frame, best_bbox, conf

    def full_image_processing(self):
        # the first time, we set up parameters
        if self.FOCAL_LENGTH_PIXELS is None: self.first_time_setup_image_parameters()
        # safety check in case the new bbox is bad, use last
        if self.valid_bbox: self.curr_bbox = self.bbox_data
        # then everytime we get the distances
        if self.yes_bbox_got:
            distance_m, offset_x_m, offset_y_m, offset_x_pixels, offset_y_pixels = self.calculate_golf_ball_metrics()
            # then based on how far off we are, instruct the drone's setpoint to move that much
            self.move_drone(offset_x_pixels, offset_y_pixels)
            # if self.drone_pose_tool == 'pixel':
            #     self.move_drone_p(offset_x_pixels, offset_y_pixels)
            # elif self.drone_pose_tool == 'meter':
            #     self.move_drone_m(distance_m, offset_x_m, offset_y_m)

    def calculate_golf_ball_metrics(self):
        # unpack all the values from the bounding box and calculate the diameter
        x1, y1, x2, y2 = self.curr_bbox
        bbox_width, bbox_height = x2 - x1, y2 - y1
        # if the ball is cut off on the edges, choose the larger dimension
        if x1 <= 0 or y1 <= 0 or x2 >= self.frame_width or y2 >= self.frame_height: diameter_pixels = max(bbox_width, bbox_height)
        else: diameter_pixels = (bbox_width + bbox_height) / 2
        # compute the golf ball's center
        ball_center_x, ball_center_y = x1 + bbox_width / 2, y1 + bbox_height / 2
        image_center_x, image_center_y = self.camera_frame_center
        # compute the distance to the ball
        distance_mm = (self.REAL_DIAMETER_MM * self.FOCAL_LENGTH_PIXELS) / diameter_pixels
        distance_m = distance_mm / 1000 
        # calculate ball and image centers (in pixel coordinates)
        offset_x_pixels, offset_y_pixels = ball_center_x - image_center_x, ball_center_y - image_center_y
        offset_x_m = self.pixels_to_meters(offset_x_pixels, distance_m)
        offset_y_m = self.pixels_to_meters(offset_y_pixels, distance_m)
        return distance_m, offset_x_m, offset_y_m, offset_x_pixels, offset_y_pixels

    # def move_drone_m(self, offset_x_m, offset_y_m, step_size=0.01):
    #     vector_length = (offset_x_m ** 2 + offset_y_m ** 2) ** 0.5
    #     if vector_length == 0: return
    #     scaled_x = (offset_x_m / vector_length) * step_size
    #     scaled_y = (offset_y_m / vector_length) * step_size
    #     # update the drone's position with the scaled values
    #     self.set_position.x = self.position.x + scaled_x
    #     self.set_position.y = self.position.y + scaled_y
    #     self.set_position.z = self.desired_flight_height
    #     print("set pose:" + self.set_position)
    #     print("pose:" + self.position)
        
    def move_drone(self, offset_x_pixels, offset_y_pixels, move_size=0.005, pixel_tol=50):
        vector_length = (offset_x_pixels ** 2 + offset_y_pixels ** 2) ** 0.5
        print("moving drone triggered", vector_length)
        if vector_length <= pixel_tol: 
            print("hovering")
            return
        print("MOVE")
        scaled_x = (offset_x_pixels / vector_length) * move_size
        scaled_y = (offset_y_pixels / vector_length) * move_size
        # update the drone's position with the scaled values
        if self.testing == True:
            self.set_position.x = self.position.x + scaled_x
            self.set_position.y = self.position.y + scaled_y
            self.set_position.z = self.desired_flight_height

        # print(self.set_position, "||", self.position)
        return

    

    # def move_drone_p(self, offset_x_pixels, offset_y_pixels,  tolerance=50):
    #     # If the drone is close enough to the center, just hover
    #     if abs(offset_x_pixels) <= tolerance and abs(offset_y_pixels) <= tolerance:
    #         print("Move: Hovering")
    #         return
    #     if abs(offset_x_pixels) >= 4 * abs(offset_y_pixels):
    #         if offset_x_pixels > 0:
    #             print("Move: Right")
    #         else:
    #             print("Move: Left")
    #     elif abs(offset_y_pixels) >= 4 * abs(offset_x_pixels):
    #         if offset_y_pixels > 0:
    #             print("Move: Forward")
    #         else:
    #             print("Move: Backward")
    #     else:
    #         # Handles diagonal movements
    #         if offset_x_pixels > 0 and offset_y_pixels > 0:
    #             print("Move: Forward Right")
    #         elif offset_x_pixels > 0 and offset_y_pixels < 0:
    #             print("Move: Backward Right")
    #         elif offset_x_pixels < 0 and offset_y_pixels > 0:
    #             print("Move: Forward Left")
    #         elif offset_x_pixels < 0 and offset_y_pixels < 0:
    #             print("Move: Backward Left")

    ################################################
    # IMAGE PROCESSING HELPERS
    ################################################

    def imgmsg_to_numpy(self, ros_image):
        # helper to convert ROS image to numpy array
        bridge = CvBridge()
        return bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')

    def first_time_setup_image_parameters(self):
        self.frame_height, self.frame_width, _ = self.image_data.shape
        self.camera_frame_center = (self.frame_width / 2, self.frame_height / 2)
        self.FOCAL_LENGTH_PIXELS = ((self.FOCAL_LENGTH_MM / self.SENSOR_WIDTH_MM) * self.frame_width) / self.DOWN_SAMPLE_FACTOR
        return

    def pixels_to_meters(self, pixel_offset, distance_m):
        # Compute displacement in mm, then convert to meters
        return (distance_m / self.FOCAL_LENGTH_PIXELS) * pixel_offset

    def meters_to_pixels(self, offset_m, distance_m):
        # Convert offset to mm, then compute pixel displacement
        return (offset_m * self.FOCAL_LENGTH_PIXELS) / distance_m

    def set_pose_initial(self):
        # Put the current position into maintained position
        self.set_position.x = 0.0
        self.set_position.y = 0.0
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
    drone_pose_tool = "meter"
    node = SegDroneControlNode(test_type, drone_pose_tool)
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
