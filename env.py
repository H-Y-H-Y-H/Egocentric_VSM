import time
import pybullet_data as pd
import gym
from CADandURDF.robot_repo.control import *
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
                 ground_type="rug", rand_fiction=False, rand_torque=False, rand_pos=False,CONSTRAIN = False):

        self.name = name
        self.mode = p.POSITION_CONTROL
        self.maxVelocity = 1.5  # lx-224 0.20 sec/60degree = 5.236 rad/s
        self.sleep_time = 1. / 240

        self.n_sim_steps = 60  # 1 step = 0.00387 s -> 0.2322 s/step
        self.inner_motor_index = [0, 3, 6, 9]
        self.middle_motor_index = [1, 4, 7, 10]
        self.outer_motor_index = [2, 5, 8, 11]
        self.CONSTRAIN = CONSTRAIN
        self.original_motor_action_space = np.pi/3
        self.motor_action_space = np.asarray([np.pi / 6, np.pi / 10, np.pi / 10,
                                              np.pi / 6, np.pi / 10, np.pi / 10,
                                              np.pi / 6, np.pi / 10, np.pi / 10,
                                              np.pi / 6, np.pi / 10, np.pi / 10])

        self.motor_action_space_shift = np.asarray([0, -1.2, 1,
                                                    0.5, -1.2, 1,
                                                    -0.5, -1.2, 1,
                                                    0, -1.2, 1])
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
        self.force = 1.8
        self.robot_view_path = None
        self.data_collection = data_cll_flag
        self.ground_type = ground_type
        self.internal_f_state = []

        obs = self.reset()
        self.action_space = gym.spaces.Box(low=-np.ones(self.dof), high=np.ones(self.dof))
        self.observation_space = gym.spaces.Box(low=-np.ones_like(obs) * np.inf, high=np.ones_like(obs) * np.inf)

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
        for i in range(len(a)):
            if self.force_rand_flag == True:
                # action_tq = random.uniform(1.5, 1.8)
                action_tq = random.uniform(1.5, 2)
                p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=a[i], force=action_tq,
                                        maxVelocity=self.maxVelocity)
            else:
                p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=a[i], force=self.force,
                                        maxVelocity=self.maxVelocity)

        self.image_stream = []
        for i in range(self.n_sim_steps):
            if self.render:
                # Capture Camera
                if self.camera_capture == True:
                    basePos, baseOrn = p.getBasePositionAndOrientation(self.robotid)  # Get model position
                    basePos_list = [basePos[0], basePos[1], 0.3]
                    # side view
                    # p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=75, cameraPitch=-20,
                    #                              cameraTargetPosition=basePos_list)  # fix camera onto model
                    # back view
                    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0, cameraPitch=-20,
                                                 cameraTargetPosition=[0, 1, 0])  # fix camera onto model
                time.sleep(self.sleep_time)

            # if (self.clock % 60 == 0 or self.clock % 60 == 10 or self.clock % 60 == 20) and self.robot_camera == True:

            #   S{t}  i  i  i  i  i  S{t+1}
            #     |   |  |  |  |  |  |
            #     in  in in in in in pred   vsm
            #     in  in in in in in in     already_move
            #
            if i % 10 == 0 and self.robot_camera == True:
                self.img = robo_camera(self.robotid, 12)
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
            self.img = robo_camera(self.robotid, 12)


    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        planeId = p.loadURDF("plane.urdf",
                             [random.uniform(-3, 3), random.uniform(-3, 3), 0],
                             p.getQuaternionFromEuler([0, 0, random.uniform(-np.pi, np.pi)]), useFixedBase=1
                             )

        if self.ground_type == "rug":
            textureId = p.loadTexture("C:/visual_project_data/ground/rug.jpg")
        elif self.ground_type == "rug_rand":
            r_id = random.randint(0, 999)
            textureId = p.loadTexture("C:/visual_project_data/ground/rug/%d.jpg" % r_id)
        elif self.ground_type == "grass_rand":
            r_id = random.randint(0, 999)
            textureId = p.loadTexture("C:/visual_project_data/ground/grass/%d.jpg" % r_id)
        else:
            textureId = p.loadTexture("C:/visual_project_data/ground/%s.png" % self.ground_type)

        # wall_textureId = p.loadTexture("wall_picture.png")
        # WallId_front = p.loadURDF("plane.urdf",[0,14,14], p.getQuaternionFromEuler([1.57,0,0]))
        # WallId_right = p.loadURDF("plane.urdf", [-10, 0, 14], p.getQuaternionFromEuler([1.57, 0, 1.57]))
        # WallId_left = p.loadURDF("plane.urdf", [10, 0, 14], p.getQuaternionFromEuler([1.57, 0, -1.57]))

        p.changeVisualShape(planeId, -1, textureUniqueId=textureId)
        # p.changeVisualShape(planeId, -1, rgbaColor=[1, 1, 1, 0.5])
        # p.changeVisualShape(WallId_front, -1, textureUniqueId=wall_textureId)
        # p.changeVisualShape(WallId_right, -1, textureUniqueId=wall_textureId)
        # p.changeVisualShape(WallId_left , -1, textureUniqueId=wall_textureId)
        p.changeDynamics(planeId, -1, lateralFriction=self.friction)
        robotStartPos = [0, 0, 0.2]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        self.robotid = p.loadURDF(self.urdf_path, robotStartPos, robotStartOrientation, flags=p.URDF_USE_SELF_COLLISION,
                                  useFixedBase=0)

        p.changeDynamics(self.robotid, 2, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 5, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 8, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 11, lateralFriction=self.friction)

        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        self.sub_p, self.sub_q = p.getBasePositionAndOrientation(self.robotid)
        self.sub_q = p.getEulerFromQuaternion(self.sub_q)
        self.sub_p, self.sub_q = np.array(self.sub_p), np.array(self.sub_q)


        self.i = 0
        self.dof = 12  # p.getNumJoints(self.robotid)

        for _ in range(60):
            p.stepSimulation()

        if self.robot_camera == True:
            self.img = robo_camera(self.robotid, 12)
        self.image_stream = []

        return self.get_obs()

    def step(self, a):

        if self.CONSTRAIN == True:
            a = a * self.motor_action_space + self.motor_action_space_shift
        else:
            a *= self.original_motor_action_space


        self.act(a)
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

    def terrian_init(self, which_type=0):
        # p.connect(p.GUI)
        p.setAdditionalSearchPath(pd.getDataPath())

        useProgrammatic = 0
        useTerrainFromPNG = 1
        useDeepLocoCSV = 2

        heightfieldSource = which_type

        # random.seed(10)

        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        heightPerturbationRange = 0.01
        terrain = 1
        if heightfieldSource == useProgrammatic:
            numHeightfieldRows = 256
            numHeightfieldColumns = 256
            heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns
            for j in range(int(numHeightfieldColumns / 2)):
                for i in range(int(numHeightfieldRows / 2)):
                    height = random.uniform(0, heightPerturbationRange)
                    heightfieldData[2 * i + 2 * j * numHeightfieldRows] = height
                    heightfieldData[2 * i + 1 + 2 * j * numHeightfieldRows] = height
                    heightfieldData[2 * i + (2 * j + 1) * numHeightfieldRows] = height
                    heightfieldData[2 * i + 1 + (2 * j + 1) * numHeightfieldRows] = height

            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[.03, .05, 0.1],
                                                  heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
                                                  heightfieldData=heightfieldData,
                                                  numHeightfieldRows=numHeightfieldRows,
                                                  numHeightfieldColumns=numHeightfieldColumns)
            terrain = p.createMultiBody(0, terrainShape)
            p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0.2, 1])

        p.changeDynamics(terrain, -1, lateralFriction=0.99)

        if heightfieldSource == useDeepLocoCSV:
            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[.5, .5, 0.5],
                                                  fileName="heightmaps/ground0.txt", heightfieldTextureScaling=128)
            terrain = p.createMultiBody(0, terrainShape)
            p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])

        if heightfieldSource == useTerrainFromPNG:
            terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[.1, .1, 12],
                                                  fileName="heightmaps/wm_height_out.png")
            textureId = p.loadTexture("heightmaps/gimp_overlay_out.png")
            terrain = p.createMultiBody(0, terrainShape)
            p.changeVisualShape(terrain, -1, textureUniqueId=textureId)

        p.changeVisualShape(terrain, -1, rgbaColor=[1, 1, 1, 1])
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        # p.setRealTimeSimulation(1)


if __name__ == '__main__':
    robot_idx = 0
    print("robot_idx:", robot_idx)
    name = 'V0%02d' % robot_idx
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())
    RAND_FIRCTION = False
    RAND_T = False
    RAND_P = False
    env = OpticalEnv(name,
                     robot_camera=False,
                     urdf_path="CADandURDF/robot_repo/V000_cam/urdf/V000_cam.urdf",
                     rand_fiction=RAND_FIRCTION,
                     rand_torque=RAND_T,
                     rand_pos= RAND_P,
                     CONSTRAIN=True
                     )
    env.sleep_time =1/480

    # Run Sin Gait Baseline
    para = np.loadtxt("CADandURDF/robot_repo/V000_cam/1.csv")
    noise = 0

    # Run the trajectory commands
    # best_action_file = np.loadtxt("test.csv")
    results = []
    pred = []
    GG_pos_ori = []
    gt = []

    RAND_FIRCTION = True
    RAND_T = True

    for epoch in range(10):
        print(epoch)
        fail = 0
        result = 0
        env.reset()
        action_logger = []

        pos = np.array([0, 0, 0.3])
        ori = np.zeros((3))
        robotid_visual = 0
        last_obs = np.zeros(6)

        for i in range(56):
            time1 = time.time()
            action = sin_move(i, para)
            action = np.random.normal(loc=action, scale=noise, size=None)
            action = np.clip(action, -1, 1)


            action_logger.append(action)

            # robotid_visual = p.loadURDF("robot_repo/V500/urdf/V500_body_visual.urdf",
            #                             pos, p.getQuaternionFromEuler(ori), useFixedBase=1)
            obs, r, done, _ = env.step(action)
            print(obs[5])

            extrapolated_pred = (obs[:6] - last_obs) + obs[:6]
            pred.append(extrapolated_pred)
            last_obs = np.copy(obs[:6])

            pos += obs[:3]
            ori = obs[3:6]

            # p.removeBody(robotid_visual)

            result = env.robot_location()
            GG_pos_ori.append(np.concatenate((result)))
            gt.append(obs[:6])
            # print(result, r, done)

            time2 = time.time()

            # print(0.12 - (time2 - time1))

        results.append(result[0])
    np.savetxt("test/baselines/data/pred.csv", np.asarray(pred))
    np.savetxt("test/baselines/data/rob_pos_ori.csv", np.asarray(GG_pos_ori))
    np.savetxt("test/baselines/data/gt.csv", np.asarray(gt))
    # np.savetxt("sin_gait_action.csv",action_logger)
    # np.savetxt("perfect_self_model/analysis/results.csv",np.asarray(results))
