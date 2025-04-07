#!/usr/bin/env python

import rosbag
import cv2
from cv_bridge import CvBridge
import rospy

# CONFIG
bag_file = "your_file.bag"
image_topic = "/camera/image_raw"
output_video = "output.mp4"
fps = 60  # Or compute from timestamps

bridge = CvBridge()

# Open bag
bag = rosbag.Bag(bag_file, 'r')

# Get image size from the first frame
for topic, msg, t in bag.read_messages(topics=[image_topic]):
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    height, width, _ = cv_image.shape
    break

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write frames
for topic, msg, t in bag.read_messages(topics=[image_topic]):
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    video_writer.write(cv_image)

bag.close()
video_writer.release()
print("✅ Video saved as:", output_video)
