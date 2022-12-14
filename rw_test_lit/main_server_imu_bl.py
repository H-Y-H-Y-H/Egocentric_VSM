# from realsense_camera.video_capture import *
from CADandURDF.robot_repo.control import *
import socket
import numpy as np
import cv2
import os
import time


class VSM_Env():
    def __init__(self, episode_len=300, debug=0):
        # Socket Conneciton
        # MAC find WiFi IP - ipconfig getifaddr en0
        HOST = '192.168.0.100'
        # Port to listen on (non-privileged ports are > 1023)
        PORT = 2059

        print('Connected')

        # Set up Socket
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((HOST, PORT))

        print('Waiting for connection[s]...')

        # Wait for client to join socket
        self.s.listen()
        self.conn, addr = self.s.accept()
        print('Connected by: ', addr)

        # Robot State values that will be bounced with client
        self.robot_state = np.zeros(3, dtype=np.float32)
        self.imu_previous = np.zeros(3, dtype=np.float32)
        # Servo positions that will be added to robot_state when sent to client
        self.servo_cmds = np.zeros(12, dtype=np.float32)
        # Start counter for waling robot
        self.i = 0

        # maximum episode length
        self.max_len = episode_len

        self.debug = debug
        if self.debug == 1:
            cv2.namedWindow('Map', cv2.WINDOW_NORMAL)

    def reset(self):
        self.i = 0
        # Resetting the motors happens in the reset method
        self.servo_cmds = np.asarray([0,-0.865,0.865,
                                      0,-0.865,0.865,
                                      0,-0.865,0.865,
                                      0,-0.865,0.865], dtype=np.float32)
        self.conn.sendall(self.servo_cmds.tobytes())
        obs = self.get_msg_from_pi()
        return obs

    def process_action(self, cmd):
        cmd = np.asarray(cmd,dtype=np.float32)
        # Prepare and send position data to clients
        self.conn.sendall(cmd.tobytes())

    def get_msg_from_pi(self):
        # Robot state is returned by reading from the motors
        self.robot_state = np.frombuffer(self.conn.recv(1024), dtype=np.float32)
        return np.array(list(self.robot_state), dtype=np.float32)

    def step(self, cmds):
        # processes and executes command
        self.process_action(cmds)
        # Receive Robot State and position from client after action execution
        recv_msg = self.get_msg_from_pi()

        recv_msg = np.asarray([recv_msg[1],recv_msg[3],recv_msg[5]])

        print(recv_msg/180*np.pi)
        # Update counter
        self.i += 1

        # Update done value
        done =  self.i >= self.max_len

        # reward function
        r = 0
        obs = recv_msg - self.imu_previous
        self.imu_previous = np.copy(recv_msg)

        return obs, r, done, {}


# def get_img_data(ti):
#     while True:
#
#         frames = pipeline.poll_for_frames()
#         if not frames:
#             continue
#
#         color_frame = frames.get_color_frame()
#         color_image = np.asanyarray(color_frame.get_data())
#         gray = cv2.cvtColor(color_image[80:560], cv2.COLOR_BGR2GRAY)
#         resized = cv2.resize(gray, dim)
#         # print('check pixel value:',np.max(resized),np.min(resized),resized[50])
#
#         cv2.imwrite(save_path_realdata + '/img/%d.jpeg' % (ti), resized)
#
#         return resized/255



def apply_model(i,IMU,choose_a,cur_theta):
    a = sin_move(i,para=sin_para)
    A_array = np.asarray([a] * num_traj)
    scalse_noise = np.asarray([noise,
                               noise,
                               noise,
                               noise ,
                               noise,
                               noise,
                               noise,
                               noise,
                               noise,
                               noise,
                               noise,
                               noise] * num_traj).reshape((num_traj, 12))

    A_array = np.random.normal(loc=A_array, scale=scalse_noise, size=None)
    a = np.clip(A_array, -1, 1)
    if PRE_A == True:
        repeat_pre_a = np.asarray([choose_a] * num_traj)
        a = np.concatenate((repeat_pre_a, a), axis=1)
    a = torch.from_numpy(a.astype(np.float32)).to(device)

    if BLACK_IMAGE == True:
        IMGs = torch.zeros((5, 128, 128), dtype=torch.float32).to(device)

    # feed into model
    IMU_s = np.asarray([IMU]).repeat(num_traj, axis=0)
    IMU_s = torch.from_numpy(IMU_s.astype(np.float32)).to(device)
    pred_ns = model.forward(IMU_s, a)
    pred_ns_numpy = pred_ns.cpu().detach().numpy()

    # mapping normalized outputs
    for kk in range(6):
        pred_ns_numpy[:, kk] /= xx2[kk]
        pred_ns_numpy[:, kk] = np.add(pred_ns_numpy[:, kk], xx1[kk])

    # Task_select
    if TASK == "f":
        all_a_rewards = (10 * pred_ns_numpy[:, 1]- 10 * abs(cur_theta + pred_ns_numpy[:, 5]) )
    elif TASK == "l":
        # all_a_rewards = 100 * np.pi - abs((np.pi / 2 - cur_theta) - pred_ns_numpy[:, 5]) * 100
        all_a_rewards = pred_ns_numpy[:, 5]
    elif TASK == "r":
        # all_a_rewards = 100 * np.pi - abs((-np.pi / 2 - cur_theta) - pred_ns_numpy[:, 5]) * 100
        all_a_rewards = -pred_ns_numpy[:, 5]
    elif TASK == "b":
        all_a_rewards = 10 - 10*pred_ns_numpy[:, 1] - abs(pred_ns_numpy[:, 0])
    elif TASK == "move_r":
        all_a_rewards = pred_ns_numpy[:, 0]
    else:
        all_a_rewards = np.zeros(num_traj)
        greedy_select = 1

    greedy_select = int(np.argmax(all_a_rewards))

    # Select Action
    if PRE_A == True:
        choose_a = a[greedy_select, 12:]
    else:
        choose_a = a[greedy_select]

    choose_a = choose_a.cpu().detach().numpy()
    pred = pred_ns_numpy[greedy_select]

    return choose_a, pred


def apply_SinGait(ti,noise_f = False):
    action = sin_move(i, sin_para)
    if noise_f == True:
        action = np.random.normal(loc=action, scale=0.2, size=None)
        action = np.clip(action, -1, 1)

    for f in range(5):
        get_img_data(i*5+f)

    return action



if __name__ == '__main__':

    mode_num = 31
    TASK = "b"
    noise = 0.2
    PRE_A = True
    BLACK_IMAGE = False

    name = "V000"
    act_dim = 12
    state_dim = 18
    num_traj = 50

    dim = (128,128)
    # Load model
    load_trained_model_path = "../train/mode%d/best_model.pt" % mode_num
    scale_coff = np.loadtxt("../norm_dataset_V000_cam_n0.2_rug_rand.csv")
    sin_para = np.loadtxt("../CADandURDF/robot_repo/V000_cam/0.csv")

    xx1, xx2 = scale_coff[0], scale_coff[1]
    from ResNet_RNN import *
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("start", device)
    model = IMU_bl(num_classes=6, ACTIVATED_F="L").to(device)
    model.load_state_dict(torch.load(load_trained_model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()

    # Construct MAIN SERVER object
    env = VSM_Env(episode_len=50,debug=0)
    save_path_realdata = "%s_%d" % (name,mode_num)

    try:
        os.mkdir(save_path_realdata)
        os.mkdir(save_path_realdata+"/img")
    except OSError:
        pass

    # Reset environment
    obs = env.reset()

    # Keep track of time for average actions/second calculation
    time0 = time.time()

    state_info, action_info, time_info = [], [], []
    done = False
    choose_a = [0] * 12
    initial_imu = np.asarray([0.] * 3)
    initial_imu = initial_imu.squeeze()
    cur_theta = 0

    IMU = initial_imu.astype(np.float32)
    log_action = []
    log_pred = []
    log_feedback_pos = []
    # Walk the robot
    for i in range(56):
        # Timestamp
        # time_info.append(time.time() - start)

        # From visual_self-model:

        choose_a,pred  = apply_model(i, IMU, choose_a, cur_theta)
        cur_theta += pred[5]
        print("cur:",cur_theta)
        log_pred.append(pred)
        # Sin_Gait:
        # choose_a = apply_SinGait(i, noise_f = True)

        obs, r, done, _ = env.step(choose_a)
        IMU = obs
        # print(obs)
        # obs = obs[:24]

        # log_feedback_pos.append(obs)
        log_action.append(choose_a)


        time1 = time.time()
        time_used = time1 - time0
        time0 = time1
        if time_used < 0.22:
            time.sleep(0.22-time_used)
        print(i, time_used)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    print('Done')
    np.savetxt(save_path_realdata+"/action.csv",np.asarray(log_action))
    np.savetxt(save_path_realdata+"/pred.csv",np.asarray(log_pred))
    # np.savetxt(save_path_realdata+"/pos.csv",np.asarray(log_feedback_pos))

    # pipeline.stop()