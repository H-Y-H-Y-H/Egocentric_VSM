import pybullet as p
import time, math
import sys
import pybullet_data
import numpy as np
from camera_pybullet import *
urdf_path = 'CADandURDF/robot_repo/'
friction = 0.99
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
atlas = p.loadURDF(urdf_path+"atlas/atlas_v4_l.urdf", [0, 0, 0.95],
                   baseOrientation=p.getQuaternionFromEuler([0,0,np.pi/2]),useFixedBase=0)
planeId = p.loadURDF("plane.urdf", [0, 0, 0])
p.changeDynamics(planeId, -1, lateralFriction=friction)

p.setGravity(0, 0, -10)
p.changeDynamics(atlas, 23, lateralFriction=friction)
p.changeDynamics(atlas, 29, lateralFriction=friction)

control_bars = []
n_joints = p.getNumJoints(atlas)
print('Joint_num: ', p.getNumJoints(atlas))

action_space_low = [0.]*19 + [-0.4, 0.8] + [0.]*9
action_space_high= [0.]*19 + [-0.7, 1.3] + [0.]*9
neck_pitch  = 0.76 # 10
init_satate = [0,0,0,
               0,-1.45,1.6,0,0,0,0,
               neck_pitch, # 10
               0, 1.45,1.6,0,0,0,0,
               0,0.1,   #18-19
               -0.6,1.2,  -0.6,-0.1,
                0,-0.1,
               -0.6,1.2,   -0.6,0.1,
               ]

rtraj_leg0 = [-0.6,-0.7,-0.7]
rtraj_leg1 = [1.2, 1.3,1.2]
para = [-0.6,-0.7,-0.7]
# ltraj_leg0 = [-0.7,-0.7,-0.6]
# ltraj_leg1 = [1.3,1.2,1.2]


leg_id = [20,21,22,26,27,28]
for i in range(p.getNumJoints(atlas)):
    p.setJointMotorControl2(atlas, i, p.POSITION_CONTROL, init_satate[i])
    joint_info = p.getJointInfo(atlas, i)
    bar_id = p.addUserDebugParameter(joint_info[1].decode("utf-8"), -3.14, 3.14, init_satate[i])
    control_bars.append(bar_id)
    print(joint_info)

for _ in range(60):
    p.stepSimulation()
def move_altas(ti, para,T=3):


    s_action = np.zeros(6)
    #
    # s_action[0] = para[0] * np.sin(ti*np.pi*2/T+para[2]) + para[4]  # left   hind  #-0.65 +
    # s_action[1] = para[1] * np.sin(ti*np.pi*2/T+para[3])  + para[5]
    # s_action[2] = -s_action[0] - s_action[1]
    # s_action[3] = - para[0] * np.sin(ti*np.pi*2/T+para[2]) + para[4]                 #-0.65 +
    # s_action[4] = -para[1] * np.sin(ti*np.pi*2/T+para[3])  + para[5]                 #1.25 +
    #s_action[5] = -s_action[3] - s_action[4]

    priode_id = ti % 3
    priode_id_l = (ti + 1 )% 3
    s_action[0] = para[priode_id] #+ para[0] * np.sin(np.pi*2/T+para[2])  # left   hind
    s_action[1] = 0.6 - s_action[0]
    # s_action[1] = para[3:][priode_id] #+ para[1] * np.sin(np.pi*2/T+para[3])


    s_action[3] = para[priode_id_l] #+ para[0] * np.sin(np.pi*2/T+para[2])
    s_action[4] = 0.6 - s_action[3]

    # s_action[4] = para[3:][priode_id_l] #+ para[1] * np.sin(np.pi*2/T+para[3])

    s_action[2] = -s_action[0] - s_action[1]
    s_action[5] = -s_action[3] - s_action[4]


    return s_action

t = 0
manual_control = False
# p.setRealTimeSimulation(1)

link_info = p.getLinkStates(atlas,list(range(31)))
z_heights = []
for i in range(30):
    z_heights.append(p.getLinkState(atlas,i)[0][2])
print(np.argsort(z_heights))

while (1):

    if manual_control:
        joint_values = []
        for i in range(n_joints):
            joint_values.append(p.readUserDebugParameter(control_bars[i]))
    else:
        # para[0] = np.random.uniform(-0.1, 0.1)
        # para[1] = np.random.uniform(-0.1, 0.1)
        # para[2] = np.random.uniform(-np.pi, np.pi)
        # para[3] = np.random.uniform(-np.pi, np.pi)
        # para[4] = np.random.uniform(-0.6, -0.7)
        # para[5] = np.random.uniform(1.2, 1.3)

        joint_values = move_altas(t,para)


    for i in range(len(leg_id)):
        p.setJointMotorControl2(atlas, leg_id[i], controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_values[i],
                                force=400,
                                maxVelocity=10)

    atlas_camera(atlas,camera_link_idx=10)
    for i in range(60):
        p.stepSimulation()
        # time.sleep(1/480)

    t += 1






