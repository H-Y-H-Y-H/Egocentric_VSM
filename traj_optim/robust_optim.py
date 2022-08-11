from env import *
from camera_pybullet import *
from CADandURDF.robot_repo.control_V5 import *



def add_noise(env,para, s_path,noise, epoch_num=5000, step_num_each_epoch=100, num_individual=32):
    best_result = - np.inf
    para_batch = [para] * num_individual
    para_batch = np.array(para_batch)
    best_para = []

    for epoch in range(epoch_num):
        print(epoch)

        para_batch = np.random.normal(loc=para_batch, scale=noise * 2 , size=None)
        para_batch = np.clip(para_batch, -2*np.pi, 2*np.pi).astype(np.float32)
        success_time = []
        logger_flag = False


        for individual in range(num_individual):
            fail = False
            para = para_batch[individual]
            result = 0
            env.reset()


            for step in range(step_num_each_epoch):

                action = sin_move(step, para)
                action = np.random.normal(loc=action, scale=noise, size=None)
                action = np.clip(para_batch, -1, 1).astype(np.float32)

                obs, r, done, info = env.step(action)
                result += 5*obs[1] - abs(obs[3]) -abs(obs[4]) - abs(obs[5])
                # print(obs[:6])

                # Check the robot position. If it is doing crazy things, abort it.
                if env.check() == True:
                    fail = True
                    break

                # pos = env.robot_location()
                # if pos[2] < 0.1:
                #     # penalty of the stupid gait like wriggle.
                #     result *= 0.5

            if fail == False:
                if result > best_result:
                    print(epoch, result, best_result)
                    logger_flag = True
                    best_result = result
                    best_para = para
                    # np.savetxt("log%s/%d.csv"%(name,epoch),para)
                    np.savetxt(s_path + "/%d.csv" % epoch, para)

        if logger_flag == True:
            para_batch = np.asarray([best_para] * num_individual)
            if best_result > 10:
                print("good break")
                break


def play_back(env, para, noise, step_num=100):
    result = 0
    for step in range(step_num):
        action = sin_move(step, para)
        action = np.random.normal(loc=action, scale=noise, size=None)
        action = np.clip(action,-1,1).astype(np.float32)
        obs, r, done, info = env.step(action)
        result += obs[1]
        if done == True:

            print("Shit!!!!!!!!!!!!!!!!!!!!!!")
            break
        print(step, "current step r: ", obs[1], "accumulated_r: ", result)
        print("state:", env.robot_location())


def trajectory_optimization(env,save_path, para=None, Train=False, noise=0.0):
    # Search Parameters
    if Train:
        env.camera_capture = False
        env.robot_camera = False

        # init_robot_stand_up()
        add_noise(env, para,noise, save_path)
    else:
        env.render = True
        env.robot_camera = True
        play_back(env, para, noise)


if __name__ == '__main__':
    robot_idx = 0
    TRAIN = False
    p.connect(p.GUI)

    noise = 0

    name = 'V%03d' % robot_idx
    print(name)
    if TRAIN:
        sin_para = np.loadtxt("dataset/control_para/BEST.csv")
    else:
        sin_para = np.loadtxt("dataset/upgraded_para/36.csv")

    env = OpticalEnv(name, robot_camera=False, camera_capture = False, urdf_path="../CADandURDF/robot_repo/%s/urdf/%s.urdf" % (name, name))
    env.sleep_time = 0.
    trajectory_optimization(env,
                            save_path ="dataset/control_para/",
                            para = sin_para,
                            noise = noise,
                            Train=TRAIN)
