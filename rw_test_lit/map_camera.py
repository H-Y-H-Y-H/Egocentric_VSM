from realsense_camera import pyrealsense2 as rs
import cv2
import os
from pupil_apriltags import Detector
import numpy as np
import math
import time


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
for x in range(5):
    pipeline.wait_for_frames()
at_detector = Detector(families='tagStandard41h12',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)


def image_capture(debug_flag = 0):
    for k in range(10):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("no frame")
            continue
        else:
            color_image = np.asanyarray(color_frame.get_data())
            image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            tags = at_detector.detect(image, estimate_tag_pose=False, camera_params=None, tag_size=None)
            # if debug_flag ==1:
            #     # draw the map
            #     cv2.rectangle(color_image, (50, 50), (640 - 50, 480 - 50), (255, 255, 255), 1)

            if len(tags) == 1:  # there are two robots
                # robot 1 location
                center1 = tags[0].center
                head_x1 = int(tags[0].corners[0][0] + tags[0].corners[1][0]) // 2
                head_y1 = int(tags[0].corners[0][1] + tags[0].corners[1][1]) // 2
                c_x1, c_y1 = int(center1[0]), int(center1[1])
                # robot direction
                theta1 = -math.atan2((-head_y1 + c_y1), (head_x1 - c_x1))
                if debug_flag == 1:
                    # plot the robot state
                    cv2.line(color_image, (c_x1, c_y1), (head_x1, head_y1), (0, 97, 255), 1)
                    cv2.circle(color_image, (320, 240), 10, (255, 0, 0))
                    cv2.imshow('Map', color_image)
                    return 1, [c_x1, c_y1, theta1], color_image
                else:
                    return 1, [c_x1, c_y1, theta1],0
            else:
                print("cannot see tags")
                if k <9:
                    continue
    return 0, 0, 0


if __name__ == "__main__":
    debug = 1
    if debug ==1:
        cv2.namedWindow('Map', cv2.WINDOW_NORMAL)
    while True:
        flag, info, img = image_capture(debug)
        print(info)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
