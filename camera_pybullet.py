import pybullet as p
import numpy as np
import cv2

fov, aspect, nearplane, farplane = 42, 1.0, 0.01, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)
dim = (128,128)
def robo_camera(robotId,camera_link_idx):
    # Center of mass position and orientation
    com_p, com_o, _, _, _, _ = p.getLinkState(robotId, camera_link_idx)

    com_p = np.asarray(com_p)
    rot_matrix = p.getMatrixFromQuaternion(com_o)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # Initial vectors
    init_camera_vector = (0, 1, 0)
    init_up_vector = (0, 0, 1)
    # Rotated vectors
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)
    view_matrix = p.computeViewMatrix(com_p, com_p + 1 * camera_vector, up_vector)
    img = p.getCameraImage(240, 240, view_matrix, projection_matrix)

    img = cv2.cvtColor(img[2][:,:,:3], cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(img, dim)

    return resized




