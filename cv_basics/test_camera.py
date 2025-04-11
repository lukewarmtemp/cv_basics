################################################
# Descriptions
################################################


'''
node subscribes to camera data from rgb_camera topic
node publishes message with just the image
'''


################################################
# Imports and Setup
################################################


# ros imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# image related
import cv2
from cv_bridge import CvBridge

# other imports
import time
try:
    from queue import Queue
except ModuleNotFoundError:
    from Queue import Queue
import threading

from rclpy.qos import QoSProfile, QoSReliabilityPolicy
# qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)

################################################
# Class Nodes
################################################


# threaded frame reader to get frames from camera
# class FrameReader(threading.Thread):

#     # queues to store frames
#     queues = []
#     _running = True

#     def __init__(self, camera, name):
#         super().__init__()
#         self.name = name
#         self.camera = camera

#     def run(self):
#         while self._running:
#             ret, frame = self.camera.read()
#             if ret:
#                 # push all frames into the queue
#                 for queue in self.queues:
#                     queue.put(frame)

#     def addQueue(self, queue):
#         self.queues.append(queue)

#     def getFrame(self, timeout=None):
#         queue = Queue(1)
#         self.addQueue(queue)
#         return queue.get(timeout=timeout)

#     def stop(self):
#         self._running = False


# ros2 camera node
class RGBCameraNode(Node):

    def __init__(self, cap):
        super().__init__('rgb_camera_node')
        # initiate the node
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 1) # qos_profile)
        self.bridge = CvBridge()
        #self.timer = self.create_timer(0.1, self.publish_frame)  # 10 Hz (We can go up to 120 Fps)
        self.cap = cap
        # self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        # check for camera starting
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera!")
            raise RuntimeError("Failed to open camera!")
        # self.frame_reader = FrameReader(self.cap, "FrameReader")
        # self.frame_reader.start()

        timer_period = 0.01 # 0.017 # seconds
      
        # Create the timer
        self.timer = self.create_timer(timer_period, self.timer_callback)
                     
        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()
   
    def timer_callback(self):
        """
        Callback function.
        This function gets called every 0.05 seconds.
        """
        # Capture frame-by-frame
        # This method returns True/False as well
        # as the video frame.
        ret, frame = self.cap.read()
            
        if ret == True:
            # Publish the image.
            # The 'cv2_to_imgmsg' method converts an OpenCV
            # image to a ROS 2 image message
            self.publisher_.publish(self.br.cv2_to_imgmsg(cv2.resize(frame, (128, 128))))
            # Display the message on the console
            self.get_logger().info('Publishing video frame')


    def stop(self):
        # stop frame reader and release camera
        self.frame_reader.stop()
        self.cap.release()

################################################
# Main
################################################


def main(args=None):

    cap = cv2.VideoCapture(0)  # Replace 0 with the correct device ID if necessary

    rclpy.init(args=args)
    node = RGBCameraNode(cap)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down camera node.')
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    print(cv2.getBuildInformation())

    main()


################################################
# END
################################################