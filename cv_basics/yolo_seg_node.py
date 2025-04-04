################################################
# Descriptions
################################################


'''
node subscribes to image data sent by rgb_camera_node
node computes the bounding box via yolo
node publishes message with both the image and bbox


CUSTOM MESSAGES
# custom_msgs/msg/ImageWithBboxConf.msg
std_msgs/Header header
sensor_msgs/Image image
int32[] bbox
float32 confidence

'''


################################################
# Imports and Setup
################################################


# ros imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
# from custom_msgs.msg import ImageWithBboxConf  # custom!


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
model = YOLO("/home/jetson/flyrs_ws/src/ParSight/ParSight/models/best.engine")
bridge = CvBridge()
print(torch.cuda.is_available())
print("Running on:", model.device)



################################################
# Class Nodes
################################################


class YOLOSegmentationNode(Node):

    def __init__(self):
        super().__init__('yolo_segmentation_node') 
        
        # ROS subscriber to RGB camera messages
        self.camera_subscriber = self.create_subscription(Image, '/camera/image_raw', self.frame_input_callback, 10)
        self.get_logger().info('Subscribed to Camera Input!')
        self.br = CvBridge()
        # ROS publisher of Image and BBOX data
        # self.publisher_ = self.create_publisher(ImageWithBboxConf, '/camera/image_with_bbox_conf', 1)
        
    ################################################
    # CALLBACKS
    ################################################

    def frame_input_callback(self, msg):
        # convert ROS Image to opencv format
        self.get_logger().info('Receiving video frame')
 
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(msg)
        
        self.run_yolo_segmentation(current_frame)

        # Display image
        # cv2.imshow("camera", current_frame)
        
        cv2.waitKey(1)
        return

    def run_yolo_segmentation(self, frame):
        # apply a confidence threshold
        # print(frame)
        start_time = time.time()
        results = model(frame, imgsz=640, conf=0.4)
        
        # initialize variables for the highest confidence detection
        best_conf, best_bbox = 0, None
        # cycle through all found bboxes
        for result in results:
            for det in result.boxes.data:
                # extract the bounding box
                x_min, y_min, x_max, y_max, conf, cls = det.tolist()
                conf = float(conf)
                # filter low confidence out
                if conf < 0.4: continue
                # check if this is the highest confidence so far
                if conf > best_conf:
                    best_conf = conf
                    best_bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        
        # if no valid bounding box found, set it to [-1, -1, -1, -1]
        if best_bbox is not None:
            # draw the bounding box on the frame
            cv2.rectangle(frame, (best_bbox[0], best_bbox[1]), (best_bbox[2], best_bbox[3]), (0, 255, 0), 2)
            label = f'Conf: {best_conf:.2f}'
            cv2.putText(frame, label, (best_bbox[0], best_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            best_bbox = [-1, -1, -1, -1]

        end_time = time.time()
        total_time = end_time - start_time
        print(total_time, best_bbox)
        # create message with the best bbox
        # self.create_msg_and_publish(frame, best_bbox, best_conf)
        return

    # def create_msg_and_publish(self, frame, bbox, conf):
    #     # convert frame to ROS Image message
    #     image_msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
    #     # create the custom message
    #     msg = ImageWithBboxConf()
    #     msg.header.stamp = self.get_clock().now().to_msg() 
    #     msg.header.frame_id = 'frame' # "map"
    #     msg.image = image_msg
    #     msg.bbox = bbox
    #     msg.confidence = conf
    #     # send the message
    #     self.publisher_.publish(msg)
    #     self.get_logger().info(f"Published image with bbox: {bbox} and confidence: {conf}")
    #     return


################################################
# MAIN EXECUTION
################################################


def main(args=None):
    rclpy.init(args=args)
    yolo_segmentation_node = YOLOSegmentationNode()   
    try:
        rclpy.spin(yolo_segmentation_node)
    except KeyboardInterrupt:
        yolo_segmentation_node.get_logger().info('Shutting down segmentation node.')
    finally:
        yolo_segmentation_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()


################################################
# END
################################################