## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################
import sys
import pyrealsense2 as rs
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import cv2
import time
import os

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
# config.enable_stream(rs.stream.color,640 , 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.color,320 , 240, rs.format.bgr8, 60)

# Start streaming
# pipeline.start(config)
#
profile = pipeline.start(config)
# depth_sensor = profile.get_device().first_depth_sensor()
# depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)

times = 0
flag = 0
photo = 0
last_image = 0

# save_path = "G:/My Drive/projects/visual_sm1/realsense_camera/test"
save_path = "C:/Users/yuhan/Desktop/visual_self-model/rw_test_lit/realsense_camera/test"
try:
    os.mkdir(save_path)
except OSError:
    pass
dim = (128,128)
last_time = time.time()


if __name__ == "__main__":
    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            # if not depth_frame or not color_frame:
            #     continue
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            # gray = cv2.cvtColor(color_image[80:560], cv2.COLOR_BGR2GRAY)
            # resized = cv2.resize(gray, dim)
            # Show images
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', color_image)


            times+=1
            time_used = time.time()-last_time
            last_time = time.time()
            print(time_used)
            cv2.imwrite(save_path + '/%d.jpeg' % (times), gray[96:(128+96),56:(128+56)])
            # cv2.imwrite(save_path + '/%d.jpeg' % (times), resized)

            # cv2.waitKey(1)
    finally:
        # Stop streaming
        pipeline.stop()
