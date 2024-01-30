import time
import pybullet_data as pd
import gym
from camera_pybullet import *
import numpy as np
import scipy.linalg as linalg
import random
import cv2
import math as m




def Rz_2D(theta):
    return np.array([[m.cos(theta), -m.sin(theta), 0],
                     [m.sin(theta), m.cos(theta), 0]])


def World2Local(pos_base, ori_base, pos_new):
    psi, theta, phi = ori_base
    R = Rz_2D(phi)
    R_in = R.T
    pos = np.asarray([pos_base[:2]]).T
    R = np.hstack((R_in, np.dot(-R_in, pos)))
    pos2 = list(pos_new[:2]) + [1]
    vec = np.asarray(pos2).T
    local_coo = np.dot(R, vec)
    return local_coo


class OpticalEnv(gym.Env):
    def __init__(self, name, robot_camera=False, camera_capture=False, data_cll_flag=False, urdf_path='CAD2URDF',
                 ground_type="rug", rand_fiction=False, rand_torque=False, rand_pos=False,):

        self.name = name
        self.mode = p.POSITION_CONTROL
        self.maxVelocity = 10
        self.sleep_time = 0


        self.camera_id = 10
        self.n_sim_steps = 60  # 1 step = 0.00387 s -> 0.2322 s/step
        self.leg_id = [20,21,22,26,27,28]

        self.log_state = [0, 0, 0, 0, 0, 0]
        self.urdf_path = urdf_path
        self.camera_capture = camera_capture
        self.counting = 0
        self.robot_camera = robot_camera
        if rand_fiction == True:
            self.friction = random.uniform(0.5, 0.99)
            # print("F:",self.friction)
        else:
            self.friction = 0.99
        self.pos_rand_flag = rand_pos
        self.force_rand_flag = rand_torque
        self.force = 400
        self.robot_view_path = None
        self.data_collection = data_cll_flag
        self.ground_type = ground_type
        self.internal_f_state = []
        self.ov_input = []
        self.init_state = [0,0,0,
               0,-1.45,1.6,0,0,0,0,
               0.76, # 10
               0, 1.45,1.6,0,0,0,0,
               0,0.1,   #18-19
               -0.6,1.2,  -0.6,-0.1,
                0,-0.1,
               -0.6,1.2,   -0.6,0.1,
               ]

        obs = self.reset()
        # self.action_space = gym.spaces.Box(low=-np.ones(self.dof), high=np.ones(self.dof))
        # self.observation_space = gym.spaces.Box(low=-np.ones_like(obs) * np.inf, high=np.ones_like(obs) * np.inf)

        p.setAdditionalSearchPath(pd.getDataPath())

    def coordinate_transform(self, input_state):
        # input_state[:3] -= self.log_state[:3]
        # print("transformation",input_state)
        radian = - self.log_state[5]
        # print("theta",theta)
        matrix_r = linalg.expm(
            np.cross(np.eye(3), [0, 0, 1] / linalg.norm([0, 0, 1]) * radian))  # [0,0,1] rotate z axis
        pos = np.asarray([input_state[0], input_state[1], input_state[2]])
        output_state = np.dot(matrix_r, pos)
        output_state = list(output_state[:3]) + list(input_state[3:5]) + [input_state[5] - self.log_state[5]]
        return output_state

    def get_obs(self):

        self.last_p = self.p
        self.last_q = self.q

        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        # self.v_p = self.p - self.last_p
        self.v_p = World2Local(self.last_p, self.last_q, self.p)
        self.v_p[2] = self.p[2] - self.last_p[2]

        self.v_q = self.q - self.last_q

        # correct the values if over 180 to -180.
        if self.v_q[2] > 1.57:
            self.v_q[2] =self.q[2] - self.last_q[2] - 2*np.pi
        elif self.v_q[2] < -1.57:
            self.v_q[2] = (2*np.pi+self.q[2]) - self.last_q[2]


        jointInfo = [p.getJointState(self.robotid, i) for i in range(12)]
        jointVals = np.array([[joint[0]] for joint in jointInfo]).flatten()

        # body_pos:3 + body_orientation:3 + joints: 12
        # normalization [-1,1];  mean:0 std 1
        obs = np.concatenate([self.v_p, self.v_q, jointVals[:12]])
        return obs

    def get_sub_obs(self):
        self.last_sub_p = self.sub_p
        self.last_sub_q = self.sub_q

        self.sub_p, self.sub_q = p.getBasePositionAndOrientation(self.robotid)
        self.sub_q = p.getEulerFromQuaternion(self.sub_q)
        self.sub_p, self.sub_q = np.array(self.sub_p), np.array(self.sub_q)

        # self.v_p = self.p - self.last_p
        self.v_sub_p = World2Local(self.last_sub_p, self.last_sub_q, self.sub_p)
        self.v_sub_p[2] = self.v_sub_p[2] - self.last_sub_p[2]

        self.v_sub_q = self.sub_q - self.last_sub_q

        # correct the values if over 180 to -180.
        if self.v_sub_q[2] > 1.57:
            self.v_sub_q[2] = self.sub_q[2] - self.last_sub_q[2] - 2 * np.pi
        elif self.v_sub_q[2] < -1.57:
            self.v_sub_q[2] = (2 * np.pi + self.sub_q[2]) - self.last_sub_q[2]

        sub_obs = np.concatenate([self.v_sub_p, self.v_sub_q])

        return sub_obs

    def act(self, a):
        if self.pos_rand_flag == True:
            a = np.random.normal(loc=a, scale=0.05, size=None)
        for a_i in range(len(self.leg_id)):
            if self.force_rand_flag == True:
                action_tq = random.uniform(-50,50) + self.force
                p.setJointMotorControl2(self.robotid, self.leg_id[a_i], controlMode=self.mode, targetPosition=a[a_i], force=action_tq,
                                        maxVelocity=self.maxVelocity)
            else:
                p.setJointMotorControl2(self.robotid, self.leg_id[a_i], controlMode=self.mode, targetPosition=a[a_i], force=self.force,
                                        maxVelocity=self.maxVelocity)

        self.image_stream = []
        self.ov_input = []
        for i in range(self.n_sim_steps):
            # if self.render:
            #     # Capture Camera
            #     if self.camera_capture == True:
            #         # side view
            #         # basePos, baseOrn = p.getBasePositionAndOrientation(self.robotid)  # Get model position
            #         # basePos_list = [basePos[0], basePos[1], 0.3]
            #         # p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=75, cameraPitch=-20,
            #         #                              cameraTargetPosition=basePos_list)  # fix camera onto model
            #         # back view
            #         p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0, cameraPitch=-20,
            #                                      cameraTargetPosition=[0, 1, 0])  # fix camera onto model
            #     time.sleep(self.sleep_time)

            # if (self.clock % 60 == 0 or self.clock % 60 == 10 or self.clock % 60 == 20) and self.robot_camera == True:

            #   S{t}  i  i  i  i  i  S{t+1}
            #     |   |  |  |  |  |  |
            #     in  in in in in in pred   vsm
            #     in  in in in in in in     vo
            #
            if i % 10 == 0 and self.robot_camera:
                self.img = atlas_camera(self.robotid, self.camera_id)
                self.ov_input.append(self.img)
                if i != 0:
                    self.image_stream.append(self.img)

                # Robot Camera
                # self.img[(self.clock % 60)//10] = robo_camera(self.robotid, 12)[:,:,:3]
                if self.robot_view_path != None:
                    sub_state_d = self.get_sub_obs()
                    self.internal_f_state.append(sub_state_d)
                    cv2.imwrite(self.robot_view_path + "%d.png" % self.counting, self.img)
                self.counting += 1
            p.stepSimulation()

        if self.robot_camera == True:
            self.img = atlas_camera(self.robotid, self.camera_id)
            self.ov_input.append(self.img)


    def reset(self):
        p.resetSimulation()

        self.robotid = p.loadURDF(self.urdf_path, [0, 0, 0.95],
                           baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=0)
        planeId = p.loadURDF("plane.urdf",
                             [random.uniform(-3, 3), random.uniform(-3, 3), 0],
                             p.getQuaternionFromEuler([0, 0, random.uniform(-np.pi, np.pi)]), useFixedBase=1
                             )
        p.changeDynamics(planeId, -1, lateralFriction=self.friction)

        p.setGravity(0, 0, -10)
        p.changeDynamics(self.robotid, 23, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 29, lateralFriction=self.friction)
        for i in range(p.getNumJoints(self.robotid)):
            p.setJointMotorControl2(self.robotid, i, p.POSITION_CONTROL, self.init_state[i])
        for _ in range(60):
            p.stepSimulation()


        textureId = p.loadTexture("CADandURDF/rug.jpg")
        p.changeVisualShape(planeId, -1, textureUniqueId=textureId)

        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        self.sub_p, self.sub_q = p.getBasePositionAndOrientation(self.robotid)
        self.sub_q = p.getEulerFromQuaternion(self.sub_q)
        self.sub_p, self.sub_q = np.array(self.sub_p), np.array(self.sub_q)


        self.i = 0
        self.dof = 12  # p.getNumJoints(self.robotid)

        if self.robot_camera == True:
            self.img = atlas_camera(self.robotid, self.camera_id)
        self.image_stream = []

        return self.get_obs()

    def step(self, action):

        self.act(action)
        obs = self.get_obs()
        pos, _ = self.robot_location()
        r = 100 * obs[1] - 50 * np.abs(obs[0])
        done = self.check()
        return obs, r, done, {}

    def robot_location(self):
        #     if call this, the absolute location of robot body will back
        position, orientation = p.getBasePositionAndOrientation(self.robotid)
        orientation = p.getEulerFromQuaternion(orientation)

        return position, orientation

    def check(self):
        pos, ori = self.robot_location()

        if self.data_collection == False:
            if abs(pos[0]) > 0.5:
                abort_flag = True
            elif abs(ori[0]) > np.pi / 6 or abs(ori[1]) > np.pi / 6:
                abort_flag = True
            elif abs(pos[2]) < 0.1:
                abort_flag = True
            else:
                abort_flag = False
        else:
            if (abs(ori[0]) > np.pi / 6 or abs(ori[1]) > np.pi / 6):
                abort_flag = True
            # elif abs(pos[2]) < 0.18:
            #     abort_flag = True
            else:
                abort_flag = False
        return abort_flag

def move_altas(ti, para,T=3):
    priode_id = ti % T
    priode_id_l = (ti + 1) % T

    if len(para) == 3:
        s_action = np.zeros(6)
        s_action[0] = para[priode_id] #+ para[0] * np.sin(np.pi*2/T+para[2])  # left   hind
        s_action[1] = 0.6 - s_action[0]
        s_action[3] = para[priode_id_l] #+ para[0] * np.sin(np.pi*2/T+para[2])
        s_action[4] = 0.6 - s_action[3]

        s_action[2] = -s_action[0] - s_action[1]
        s_action[5] = -s_action[3] - s_action[4]
    else:
        s_action = np.zeros((len(para),6))
        s_action[:,0] = para[:,priode_id]
        s_action[:,1] = 0.6 - s_action[:,0]

        s_action[:,3] = para[:,priode_id_l]  # + para[0] * np.sin(np.pi*2/T+para[2])
        s_action[:,4] = 0.6 - s_action[:,3]

        s_action[:,2] = -s_action[:,0] - s_action[:,1]
        s_action[:,5] = -s_action[:,3] - s_action[:,4]

    # s_action[0] = para[0] * np.sin(ti*np.pi*2/T+para[2]) + para[4]  # left   hind  #-0.65 +
    # s_action[1] = para[1] * np.sin(ti*np.pi*2/T+para[3])  + para[5]
    # s_action[2] = -s_action[0] - s_action[1]
    #
    # s_action[3] = - para[0] * np.sin(ti*np.pi*2/T+para[2]) + para[4]                 #-0.65 +
    # s_action[4] = -para[1] * np.sin(ti*np.pi*2/T+para[3])  + para[5]                 #1.25 +
    # s_action[5] = -s_action[3] - s_action[4]

    return s_action

# def batch_random_para_atlas(para_batch,Gaussian = False):
#     if Gaussian == False:
#         for i in range(6):
#             if i == 4:
#                 para_batch[i][i] = random.uniform(-0.6, -0.7)
#             elif i ==5:
#                 para_batch[i][i] = random.uniform(1.2, 1.3)
#             elif i in [2,3]:
#                 para_batch[i][i] *= 2*np.pi
#             else:
#                 para_batch[i][i] = random.uniform(-0.1, 0.1)
#
#     else:
#         for i in range(6):
#             para_batch[i][i] = np.random.normal(para_batch[i][i],scale = 0.1)
#     return para_batch
def batch_random_para_atlas(para_batch,Gaussian = False):
    if Gaussian == False:
        for i in range(3):
            para_batch[i][i] += random.uniform(-0.1, 0.1)

    else:
        for i in range(3):
            para_batch[i][i] = np.random.normal(para_batch[i][i],scale = 0.1)
    return para_batch

def rand_para_gss(para_batch):
    return np.random.normal(para_batch,0.1)


if __name__ == '__main__':
    name = 'atlas'
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())
    RAND_FIRCTION = False
    RAND_T = False
    RAND_P = False
    env = OpticalEnv(name,
                     robot_camera=True,
                     urdf_path="CADandURDF/robot_repo/atlas/atlas_v4_l.urdf",
                     rand_fiction=RAND_FIRCTION,
                     rand_torque=RAND_T,
                     rand_pos= RAND_P,
                     )
    env.sleep_time =0

    n_joints = 30
    manual_control = False
    t = 0
    leg_id = [20, 21, 22, 26, 27, 28]

    while (1):
        para = [-0.6,-0.7,-0.7]

        for step_i in range(56):
            action = move_altas(step_i, para)
            env.step(action)


        t += 1

    # # Run Sin Gait Baseline
    # para = np.loadtxt("CADandURDF/robot_repo/V000_cam/1.csv")
    # noise = 0

    # Run the trajectory commands
    # best_action_file = np.loadtxt("test.csv")
    # results = []
    # pred = []
    # GG_pos_ori = []
    # gt = []
    #
    # RAND_FIRCTION = True
    # RAND_T = True
    #
    # for epoch in range(10):
    #     print(epoch)
    #     fail = 0
    #     result = 0
    #     env.reset()
    #     action_logger = []
    #
    #     pos = np.array([0, 0, 0.3])
    #     ori = np.zeros((3))
    #     robotid_visual = 0
    #     last_obs = np.zeros(6)
    #
    #     for i in range(56):
    #         time1 = time.time()
    #         action = sin_move(i, para)
    #         action = np.random.normal(loc=action, scale=noise, size=None)
    #         action = np.clip(action, -1, 1)
    #
    #
    #         action_logger.append(action)
    #
    #         # robotid_visual = p.loadURDF("robot_repo/V500/urdf/V500_body_visual.urdf",
    #         #                             pos, p.getQuaternionFromEuler(ori), useFixedBase=1)
    #         obs, r, done, _ = env.step(action)
    #         print(obs[5])
    #
    #         extrapolated_pred = (obs[:6] - last_obs) + obs[:6]
    #         pred.append(extrapolated_pred)
    #         last_obs = np.copy(obs[:6])
    #
    #         pos += obs[:3]
    #         ori = obs[3:6]
    #
    #         # p.removeBody(robotid_visual)
    #
    #         result = env.robot_location()
    #         GG_pos_ori.append(np.concatenate((result)))
    #         gt.append(obs[:6])
    #         # print(result, r, done)
    #
    #         time2 = time.time()
    #
    #         # print(0.12 - (time2 - time1))
    #
    #     results.append(result[0])
    # np.savetxt("test/baselines/data/pred.csv", np.asarray(pred))
    # np.savetxt("test/baselines/data/rob_pos_ori.csv", np.asarray(GG_pos_ori))
    # np.savetxt("test/baselines/data/gt.csv", np.asarray(gt))
    # # np.savetxt("sin_gait_action.csv",action_logger)
    # # np.savetxt("perfect_self_model/analysis/results.csv",np.asarray(results))
