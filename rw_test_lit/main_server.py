from realsense_camera.video_capture import *
from CADandURDF.robot_repo.control import *
import socket
import numpy as np
import cv2
import os


class VSM_Env():
    def __init__(self, episode_len=300, debug=0):
        # Socket Conneciton
        # MAC find WiFi IP - ipconfig getifaddr en0
        HOST = '192.168.0.225'
        # Port to listen on (non-privileged ports are > 1023)
        PORT = 8888

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
        self.robot_state = None
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
        self.robot_state = np.frombuffer(self.conn.recv(128), dtype=np.float32)
        return np.array(list(self.robot_state), dtype=np.float32)

    def step(self, cmds):
        # processes and executes command
        self.process_action(cmds)
        # Receive Robot State and position from client after action execution
        recv_msg = self.get_msg_from_pi()

        # Update counter
        self.i += 1

        # Update done value
        done =  self.i >= self.max_len

        # reward function
        r = 0

        return recv_msg, r, done, {}


def get_img_data(ti):
    while True:
        frames = pipeline.poll_for_frames()
        if not frames:
            continue

        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image[80:560], cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, dim)
        # print('check pixel value:',np.max(resized),np.min(resized),resized[50])

        cv2.imwrite(save_path_realdata + '/img/%d.jpeg' % (ti), resized)

        return resized/255



def apply_model(i,IMGs,choose_a,cur_theta):
    a = sin_move(i,para=sin_para)
    A_array = np.asarray([a] * num_traj)
    scalse_noise = np.asarray([noise,
                               noise,
                               noise,
                               noise,
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
    IMGs = IMGs.repeat(num_traj,1,1,1)
    print(IMGs.shape)
    pred_ns = model.forward(IMGs, a)
    pred_ns_numpy = pred_ns.cpu().detach().numpy()

    # mapping normalized outputs
    for kk in range(6):
        pred_ns_numpy[:, kk] /= xx2[kk]
        pred_ns_numpy[:, kk] = np.add(pred_ns_numpy[:, kk], xx1[kk])

    # Task_select
    if TASK == "f":
        # all_a_rewards = (10 * pred_ns_numpy[:, 1] - 2 * abs(cur_theta + pred_ns_numpy[:, 5]) )
        all_a_rewards = (10 * pred_ns_numpy[:, 1] - 2 * abs( pred_ns_numpy[:, 5]))
        # all_a_rewards = (10 * pred_ns_numpy[:, 1] + 5 *(pred_ns_numpy[:, 5]) - 1*(pred_ns_numpy[:, 0]))
    elif TASK == "l":
        all_a_rewards = 100 * np.pi - abs((np.pi / 2 - cur_theta) - pred_ns_numpy[:, 5]) * 100
        # all_a_rewards = 10 - (np.pi / 2 - pred_ns_numpy[:, 5]) ** 2

    elif TASK == "r":
        # all_a_rewards = 10 - (-np.pi / 2 - pred_ns_numpy[:, 5]) ** 2
        all_a_rewards = 100 * np.pi - abs((-np.pi / 2 - cur_theta) - pred_ns_numpy[:, 5]) * 100

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



    IMGs = []
    for f in range(5):
        IMGs.append(get_img_data(i*6+f))
        print("f%d" % f, end=" ")
    get_img_data(i * 6 + 5)
    print("f")

    IMGs = np.asarray(IMGs)
    IMGs = torch.from_numpy(IMGs.astype(np.float32)).to(device)

    return IMGs, choose_a, pred


def apply_SinGait(ti,noise_f = False):
    action = sin_move(ti, sin_para)
    if noise_f == True:
        action = np.random.normal(loc=action, scale=0.2, size=None)
        action = np.clip(action, -1, 1)

    for f in range(5):
        get_img_data(ti*5+f)
    return action

def rw_data_collection(steps = 1000):
    S, A, NS, DONE = [], [], [], []
    ti = 0
    time0 = time.time()
    for i in range(steps):

        print(i)
        a = sin_move(ti,para=sin_para)
        a = np.random.normal(loc=a, scale=noise, size=None)
        a = np.clip(a, -1, 1)
        obs, r, done, _ = env.step(a)
        IMGs = []

        for f in range(5):
            IMGs.append(get_img_data(i * 6 + f))
        get_img_data(i * 6 + 5)


        A.append(a)


        ti += 1



        np.savetxt(save_path_realdata + 'A.csv', np.asarray(A).astype(np.float32))
        np.savetxt(save_path_realdata + 'DONE.csv', np.array(DONE).astype(np.float32))

        time1 = time.time()
        time_used = time1 - time0
        time0 = time1
        if time_used < 0.22:
            time.sleep(0.22-time_used)



if __name__ == '__main__':

    model_type = "103_broken_recovery_blur"
    # model_type = "103"
    TASK = "l"
    noise = 0.2
    PRE_A = True
    BLACK_IMAGE = False

    name = "V000"
    act_dim = 12
    state_dim = 18
    num_traj = 50

    dim = (128,128)
    # Load model
    load_trained_model_path = "../train/model%s/best_model.pt" % model_type
    scale_coff = np.loadtxt("../norm_dataset_V000_cam_n0.2_mix4.csv")
    sin_para = np.loadtxt("../CADandURDF/robot_repo/V000_cam/0.csv")

    xx1, xx2 = scale_coff[0], scale_coff[1]
    from ResNet_RNN import *
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("start", device)
    model = ResNet50(ACTIVATED_F='L',img_channel=5, num_classes=6, input_pre_a=PRE_A,normalization=True).to(device)
    model.load_state_dict(torch.load(load_trained_model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()

    # Construct MAIN SERVER object
    env = VSM_Env(episode_len=50,debug=0)
    obs = env.reset()


    # Test model
    save_path_realdata = "%s_%s" % (name,model_type)
    try:
        os.mkdir(save_path_realdata)
        os.mkdir(save_path_realdata+"/img")
    except OSError:
        pass



    # Keep track of time for average actions/second calculation
    time0 = time.time()

    state_info, action_info, time_info = [], [], []
    done = False
    choose_a = [0] * 12
    initial_img = get_img_data(0)
    initial_img = np.asarray([initial_img] * 5)
    initial_img = initial_img.squeeze()
    cur_theta = 0


    IMGs = torch.from_numpy(initial_img.astype(np.float32)).to(device)
    log_action = []
    log_pred = []
    log_feedback_pos = []

    for i in range(56):
        # From visual_self-model:
        IMGs, choose_a, pred = apply_model(i, IMGs, choose_a, cur_theta)
        cur_theta += pred[5]
        print("cur:",cur_theta)
        log_pred.append(pred)
        # Sin_Gait:
        # choose_a = apply_SinGait(i, noise_f = True)
        obs, r, done, _ = env.step(choose_a)

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
    pipeline.stop()


    # Data Collection in the real wrold
    # datasetID = '44'
    # save_path_realdata ="rw_data/%s/" % (datasetID)
    # try:
    #     os.mkdir(save_path_realdata)
    #     os.mkdir(save_path_realdata+"/img")
    # except OSError:
    #     pass
    # rw_data_collection()
    # pipeline.stop()

