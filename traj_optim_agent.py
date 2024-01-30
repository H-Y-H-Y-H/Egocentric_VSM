import os
import numpy as np
from env_agent import *
from camera_pybullet import *



def hill_climber(env,para, s_path, epoch_num=5000, step_num_each_epoch=60, num_individual=6):

    best_result = - np.inf
    para_batch = [para] * num_individual
    para_batch = np.array(para_batch)
    best_para = []

    for epoch in range(epoch_num):
        print(epoch)
        # Random 16 individuals with specific parameter region.
        para_batch = batch_random_para_atlas(para_batch,Gaussian=True)
        # Random all parameters for the rest 4 robots.
        for i in range(3, num_individual-1):
            para_batch[i] = rand_para_gss(para_batch[i])

        result_list = []
        logger_flag = False
        for individual in range(num_individual):
            result = 0
            para = para_batch[individual]
            # Each individual will be test 100 times.
            fail = False
            for test_time in range(5):
                if fail:
                    result = -1000
                    break
                obs = env.reset()
                last_action = np.zeros(6)
                for step in range(step_num_each_epoch):
                    para_tuned = np.random.normal(loc=para, scale=noise, size=None)

                    action = move_altas(step, para_tuned)

                    obs, r, done, info = env.step(action)
                    robo_pos, robot_ori = env.robot_location()

                    last_action = np.copy(action)

                    # Check the robot position. If it is doing crazy things, abort it.
                    if env.check() == True:
                        fail = True
                        result =-1000
                        break

                    # result +=    obs[1]*50  \
                    #           - 20*abs(obs[3]) \
                    #           - 20*abs(obs[4]) \
                    #           - 20*abs(obs[0]) \
                    #          - 20 * abs(obs[2])\
                    #           - 5*abs(robo_pos[2]-0.1629)
                    result += 2*robo_pos[1] - abs(robo_pos[0]) - 0.5*abs(robot_ori[2])

            result_list.append(result)

        # print(result_list)
        best_id = np.argmax(result_list)
        best_para = para_batch[best_id]
        if result_list[best_id] > best_result:
            best_result = result_list[best_id]
            print(epoch, best_result)
            logger_flag = True
            # np.savetxt("log%s/%d.csv"%(name,epoch),para)
            np.savetxt(s_path + "/%d.csv" % epoch, best_para)
        para_batch = np.asarray([best_para] * num_individual)



# def init_robot_stand_up(cam=True):
#     for i in range(1000):
#         # action_init is compute by hand to make sure the robot can stay still.
#         action_init = [0, 0.8, -0.8,
#                        0.2, -0.5, 0.5,
#                        -0.2, -0.5, 0.5,
#                        0, 0.8, -0.8]
#         init_state_para = compute_init_para(action_init)
#         action = np.asarray(action_init, dtype="float64")
#         img = robo_camera(env.robotid, 12)
#         # plt.imsave("dataset/%05d.png"%i,img)
#         obs, r, done, _ = env.step(action)


def play_back(env, para, noise,epoch_num, step_num=28):
    result = 0
    for i in range(epoch_num):
        env.reset()
        for step in range(step_num):
            tuned_para = np.random.normal(loc=para, scale=noise, size=None)
            action = move_altas(step, tuned_para)

            # action = np.random.normal(loc=action, scale=[noise * 2,
            #                                    noise,
            #                                    noise,
            #                                    noise * 2,
            #                                    noise,
            #                                    noise,
            #                                    noise * 2,
            #                                    noise,
            #                                    noise,
            #                                    noise * 2,
            #                                    noise,
            #                                    noise], size=None)

            # action = np.ones(12) * ((-1) ** (step//20))
            obs, r, done, info = env.step(action)
            result += obs[1]
            if done == True:

                print("Shit!!!!!!!!!!!!!!!!!!!!!!")
                break
            print(step, "current step r: ", obs[1], "accumulated_r: ", result)
            print("state:", env.robot_location())


def trajectory_optimization(env,save_path,render_flag = None, para=None, Train=False, noise=0.0):
    # Search Parameters
    if Train:
        if render_flag == True:
            env.render = True
        env.camera_capture = False
        env.robot_camera = False

        # init_robot_stand_up()
        hill_climber(env, para, save_path)
    else:
        env.render = True
        env.robot_camera = True
        play_back(env, para, noise, epoch_num = 10,step_num=40)


if __name__ == '__main__':
    name = 'atlas'
    log_root = "traj_optim/dataset%s/"%name
    TRAIN = False

    num_channel = 16 #6,13
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    noise = 0.01

    print(name)
    if TRAIN:
        # para = np.zeros(6)
        # para[0] = np.random.uniform(-0.1, 0.1)
        # para[1] = np.random.uniform(-0.1, 0.1)
        # para[2] = np.random.uniform(-np.pi, np.pi)
        # para[3] = np.random.uniform(-np.pi, np.pi)
        # para[4] = np.random.uniform(-0.6, -0.7)
        # para[5] = np.random.uniform(1.2, 1.3)

        para = np.loadtxt('CADandURDF/robot_repo/atlas/ori_gait.csv')
        # sin_para = np.loadtxt(log_root+"/0_%s.csv"%name[:4])

    else:
        list_dir = os.listdir(log_root + '/control_para%d' % num_channel)
        sorted_files = sorted(list_dir, key=lambda x: int(x.split('.')[0]))
        print((sorted_files[-1]))
        para = np.loadtxt(log_root+'/control_para%d/%s'%(num_channel,sorted_files[-1]))
        # para = np.loadtxt('CADandURDF/robot_repo/atlas/control_para.csv')


    env = OpticalEnv(name,
                     robot_camera=False,
                     urdf_path="CADandURDF/robot_repo/atlas/atlas_v4_l.urdf",
                     rand_fiction=False,
                     rand_torque=False,
                     rand_pos= False,
                     )#,
                     # rand_pos= True, rand_torque= True,rand_fiction=True)
    env.sleep_time = 0.
    env.data_collection = True

    print("NUM_CHANNEL:",num_channel)
    save_path = log_root+"/control_para%d/"%num_channel
    print(save_path)
    os.makedirs(save_path,exist_ok=True)

    trajectory_optimization(env,
                            render_flag=True,
                            save_path =save_path,
                            para = para,
                            noise = noise,
                            Train=TRAIN)

