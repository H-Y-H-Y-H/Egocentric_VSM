from env import *
from camera_pybullet import *
from CADandURDF.robot_repo.control import *
import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision

# from torchsummary import summary


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("start", device)


def data_collection(env, steps, parameters, GAUSSIAN, save_path='./', noise=0.1):
    S, A, NS, DONE = [], [], [], []
    ti = 0
    fail_flag = 0
    s = env.reset()
    for i in range(steps):

        a = sin_move(ti, parameters)  # walking gait.
        if GAUSSIAN == 1:
            a = np.random.normal(loc=a, scale=[noise * 2,
                                               noise,
                                               noise,
                                               noise * 2,
                                               noise,
                                               noise,
                                               noise * 2,
                                               noise,
                                               noise,
                                               noise * 2,
                                               noise,
                                               noise], size=None)
            np.clip(a, -1, 1).astype(np.float32)
        ns, r, done, _ = env.step(a)  # ns == next state
        S.append(s)
        A.append(a)
        NS.append(ns)
        if ti == 99 or (i + 1) == steps:
            done = True
        DONE.append(done)

        s = ns
        ti += 1

        if done == True:
            s = env.reset()
            ti = 0

    S = np.array(S).astype(np.float32)
    A = np.array(A).astype(np.float32)
    NS = np.array(NS).astype(np.float32)
    DONE = np.array(DONE).astype(np.float32)

    np.savetxt(save_path + 'S.csv', S)
    np.savetxt(save_path + 'A.csv', A)
    np.savetxt(save_path + 'NS.csv', NS)
    np.savetxt(save_path + 'DONE.csv', DONE)

    data = [S, A, NS]
    print('Data Collected: ' + str(len(data[0])) + ' steps')
    return data


# Each step, read 5 images
class ImgData2(Dataset):

    def __init__(self, mode, num_data, info, data_path, transform=None, input_previous_action=False):
        self.root = data_path
        self.input_previous_action = input_previous_action
        if mode == "train":
            self.start_idx = 0
            self.end_idx = int(num_data * 0.8)
        else:
            self.start_idx = int(num_data * 0.8)
            self.end_idx = num_data

        self.all_A, self.all_S, self.all_NS, self.all_Done = info

    def __getitem__(self, idx):
        # time0 = time.time()
        pack_id = idx // 1000 + self.start_idx
        idx_data = idx % 1000
        root_dir = self.root + "%d/" % pack_id

        Done_data = self.all_Done[idx // 1000]
        Done1 = Done_data[idx_data - 1]
        Done2 = Done_data[idx_data - 2]

        img_pack = []
        if Done1 == 1 or idx_data == 0:
            pre_action = torch.zeros(12).to(device, dtype=torch.float)
        else:
            pre_action = self.all_A[idx // 1000][idx_data - 1]

        cur_state = self.all_S[idx // 1000][idx_data]
        # # time1 = time.time()
        next_state = self.all_NS[idx // 1000][idx_data]
        action = self.all_A[idx // 1000][idx_data]

        if self.input_previous_action == True:
            action = torch.concat((pre_action, action))
        sample = {'A': action, "S":cur_state, "NS": next_state}
        # time3 = time.time()
        # print("loaddata2:",time3-time0)

        return sample

    def __len__(self):
        return (self.end_idx - self.start_idx) * 1000

    def transform_img(self,img):

        if BLUR_IMG ==True:
            if random.randint(0, 1) == 1:
                sig_r_xy = random.uniform(0.1, 5)
                win_r = 2 * random.randint(1, 20) + 1
                img = cv2.GaussianBlur(img, (win_r, win_r), sigmaX=sig_r_xy, sigmaY=sig_r_xy, borderType=cv2.BORDER_DEFAULT)

            IMG = torch.from_numpy(img).to(device, dtype=torch.float)

            if random.randint(0, 1) == 1:
                T = torchvision.transforms.ColorJitter(brightness=[0.1, 10])
                IMG =T(IMG)

        else:
            IMG = torch.from_numpy(img).to(device, dtype=torch.float)

        return IMG



def process_data_Fast_Feq(data, data_range, batch_size=128, input_previous_action=False, random=True):
    A, NS, DONE = data
    A = np.array(A).astype(np.float32)
    NS = np.array(NS).astype(np.float32)
    DONE = np.array(DONE).astype(np.float32)

    Canc_IMGs, As, Ss, Ns = [], [], [], []

    print(len(DONE))

    for pack_id in range(data_range[0], data_range[1]):
        root_dir = dataset_path + "%d/" % pack_id

        for sub_idx in range(1000):
            img_pack = []
            pack_real_id = pack_id - data_range[0]

            if BLACK_IMAGE == False:
                if DONE[pack_real_id * 1000 + sub_idx - 1] == True or sub_idx == 0:
                    img_path0 = root_dir + "frames/%d.png" % (sub_idx * 6)
                    img_pack.append([plt.imread(img_path0)] * 5)
                else:
                    for i in range(1, 6):
                        img_path0 = root_dir + "frames/%d.png" % ((sub_idx - 1) * 6 + i)
                        img_pack.append(plt.imread(img_path0))
                IMG = np.asarray(img_pack)
            else:
                IMG = np.zeros((5, 128, 128)).astype(np.float32)

            # [:, :, :1]
            if len(IMG.shape) == 4:
                IMG = IMG.squeeze()

            Canc_IMGs.append(IMG)

            if input_previous_action == True:
                action_data = np.concatenate((A[pack_real_id * 1000 + sub_idx - 1], A[pack_real_id * 1000 + sub_idx]))

            else:
                action_data = A[pack_real_id * 1000 + sub_idx]

            As.append(action_data)
            Ns.append(NS[pack_real_id * 1000 + sub_idx][:6])

    As, Ns = np.array(As), np.array(Ns)
    Canc_IMGs = np.array(Canc_IMGs)
    if random:
        p = np.random.permutation(len(Canc_IMGs))
        As, Ns = As[p], Ns[p]
        Canc_IMGs = Canc_IMGs[p]

    Canc_IMGs = np.array_split(Canc_IMGs, len(Canc_IMGs) / batch_size)
    As = np.array_split(As, len(As) / batch_size)
    Ns = np.array_split(Ns, len(Ns) / batch_size)
    batches = [(Canc_IMGs[i], As[i], Ns[i]) for i in range(len(Ns)) if len(Canc_IMGs[i]) == len(As[i]) == len(Ns[i])]

    return batches


def test_model(model, env, parameters, save_flag, step_num=100, epoisde_times=10, num_traj=50, noise=0.05,
               input_pre_a=0, noralization=True):
    global TASK

    model.eval()
    if save_flag == True:
        env.robot_view_path = test_log_path + "/img/"

    log_action = []
    log_gt = []
    log_pred = []
    log_results = []
    log_done = []
    rob_pos_ori = []

    for epoch in range(epoisde_times):
        real_reward = 0
        cur_s = env.reset()[3:6]
        reward = 0

        initial_img = env.img

        initial_img = np.asarray([initial_img] * 5) / 255
        initial_img = initial_img.squeeze()

        IMGs = torch.from_numpy(initial_img.astype(np.float32)).to(device)
        choose_a = [0] *12
        cur_p = [0]*3
        cur_theta = 0
        for step in range(step_num):
                a = sin_move(step, parameters)
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

                if input_pre_a == True:
                    repeat_pre_a = np.asarray([choose_a] * num_traj)
                    a = np.concatenate((repeat_pre_a, a), axis=1)

                a = torch.from_numpy(a.astype(np.float32)).to(device)

                if BLACK_IMAGE == True:
                    IMGs = torch.zeros((5, 128, 128), dtype=torch.float32).to(device)

                if BlACK_ACTION == True:
                    a = torch.zeros_like(a)


                cur_s = np.asarray([cur_s]).repeat(num_traj,axis=0)
                cur_s = torch.from_numpy(cur_s.astype(np.float32)).to(device)
                pred_ns = model.forward(cur_s, a)

                pred_ns_numpy = pred_ns.cpu().detach().numpy()
                if REAL_OUTPUT == True:
                    if noralization == True:
                        for kk in range(2):
                            pred_ns_numpy[:, kk] /= xx2[kk]
                            pred_ns_numpy[:, kk] = np.add(pred_ns_numpy[:, kk], xx1[kk])
                        pred_ns_numpy[:, 2] /= xx2[5]
                        pred_ns_numpy[:, 2] = np.add(pred_ns_numpy[:, 2], xx1[5])
                else:
                    if noralization == True:
                        for kk in range(6):
                            pred_ns_numpy[:, kk] /= xx2[kk]
                            pred_ns_numpy[:, kk] = np.add(pred_ns_numpy[:, kk], xx1[kk])


                # Define Object Function to Compute Rewards
                # Sum up the rewards on each future step.
                # all_a_rewards = (100 * pred_ns_numpy[:,1] - 50 * np.abs(pred_ns_numpy[:,0]) - 50*np.abs(pred_ns_numpy[:,3]) )
                # if REAL_OUTPUT == True:
                #     if OPERATE == True:
                #         TASK = input()
                #     if TASK == "f":
                #         all_a_rewards = (100 * pred_ns_numpy[:, 1] - 50 * abs(pred_ns_numpy[:, 0]))
                #     elif TASK == "l":
                #         all_a_rewards = 10 - (np.pi / 2 - pred_ns_numpy[:, 2]) ** 2
                #     elif TASK == "r":
                #         all_a_rewards = 10 - (-np.pi / 2 - pred_ns_numpy[:, 2]) ** 2
                #     elif TASK == "stop":
                #         all_a_rewards = 10 - abs(pred_ns_numpy[:, 1]) - abs(pred_ns_numpy[:, 0])
                #     elif TASK == "move_r":
                #         all_a_rewards = pred_ns_numpy[:, 0]
                #     elif TASK == "b":
                #         all_a_rewards = (-50 * pred_ns_numpy[:, 1])
                #     else:
                #         all_a_rewards = np.zeros(num_traj)
                #     greedy_select = int(np.argmax(all_a_rewards))
                # else:
                if TASK == "f":
                    all_a_rewards = 10 * pred_ns_numpy[:, 1] - 10 * abs(cur_theta + pred_ns_numpy[:, 5]) -5*abs(cur_p[0]+ pred_ns_numpy[:,0])
                elif TASK == "l":
                    all_a_rewards =  pred_ns_numpy[:, 5] - abs(cur_p[1] + pred_ns_numpy[:,1]) - abs(cur_p[0] + pred_ns_numpy[:,0])
                elif TASK == "r":
                    all_a_rewards = -pred_ns_numpy[:, 5] - abs(cur_p[1] + pred_ns_numpy[:,1]) - abs(cur_p[0] + pred_ns_numpy[:,0])
                elif TASK == "stop":
                    all_a_rewards = 10 - abs(pred_ns_numpy[:, 1]) - abs(pred_ns_numpy[:, 0])
                elif TASK == "move_r":
                    all_a_rewards = pred_ns_numpy[:, 0]
                elif TASK == "b":
                    all_a_rewards = -20 * pred_ns_numpy[:, 1] - 10 * abs(cur_theta + pred_ns_numpy[:, 5])-5*abs(cur_p[0]+ pred_ns_numpy[:,0])
                else:
                    all_a_rewards = np.zeros(num_traj)
                greedy_select = int(np.argmax(all_a_rewards))
                c_pos,c_ori = env.robot_location()
                cur_x = c_pos[0]

                cur_theta = c_ori[2]


                if input_pre_a == True:
                    choose_a = a[greedy_select, 12:]
                else:
                    choose_a = a[greedy_select]
                choose_a = choose_a.cpu().detach().numpy()
                pred = pred_ns_numpy[greedy_select]
                c_pos,c_ori = env.robot_location()
                # cur_x = c_pos[0]
                # cur_theta = c_ori[2]
                cur_x += pred[0]
                cur_theta += pred[5]
                print("accumulated pred:",cur_x,cur_theta)

                # Only choose the action that has largest rewards.

                obs, r_step, done, _ = env.step(choose_a)


                log_action.append(choose_a)
                print(env.robot_location())
                real_reward += r_step


                gt = obs[:6]
                cur_s = obs[3:6]

                print("PRED and GT:",pred[5],gt[5])
                log_gt.append(gt)
                log_pred.append(pred)
                rob_pos_ori.append(np.concatenate((c_pos, c_ori)))

                img = env.image_stream
                img = np.asarray(img) / 255
                IMGs = torch.from_numpy(img.astype(np.float32)).to(device)

                if step == step_num - 1:

                    log_done.append(1)
                    break
                else:
                    log_done.append(0)

        log_results.append(real_reward)

    np.savetxt(test_log_path + '/data/pred.csv', np.asarray(log_pred))
    np.savetxt(test_log_path + '/data/gt.csv', np.asarray(log_gt))
    np.savetxt(test_log_path + '/data/done.csv', np.asarray(log_done))
    np.savetxt(test_log_path + '/data/rob_pos_ori.csv', np.asarray(rob_pos_ori))
    if LOG_A == True:
        np.savetxt(test_log_path + '/data/log_action.csv', np.asarray(log_action))

    result_mean = np.mean(log_results)
    result_std = np.std(log_results)
    print("test result:", result_mean, result_std)

    return log_results


def train_in_sim(env, parameters, data_num, batchsize, train_info, valid_info, log_path, epochs, lr=1e-4, save=False,
                 plot=False, input_previous_action=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)
    training_data = ImgData2(mode="train",
                             info=train_info,
                             num_data=data_num,
                             data_path=dataset_path, input_previous_action=input_previous_action)
    validation_data = ImgData2(mode="valid",
                               info=valid_info,
                               num_data=data_num,
                               data_path=dataset_path, input_previous_action=input_previous_action)

    train_dataloader = DataLoader(training_data, batch_size=batchsize, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(validation_data, batch_size=batchsize, shuffle=True, num_workers=0)

    if pre_trained_model == True:
        model.load_state_dict(torch.load(pre_trained_model_path))

    training_L, validation_L = [], []
    min_loss = + np.inf

    abort_learning = 0
    decay_lr = 0
    t0 = time.time()

    for epoch in range(epochs):
        # Training Period
        l = []
        model.train()

        for i, batch in enumerate(train_dataloader):
            As, S, Ns = batch["A"],batch["S"], batch["NS"]

            pred = model.forward(S[:,3:6], As)
            if REAL_OUTPUT == True:
                output_d = torch.hstack((Ns[:, :2], Ns[:, 5:]))
            else:
                output_d = Ns
            loss = model.loss(pred, output_d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l.append(loss.item())

        epoch_t_loss = np.mean(l)
        training_L.append(epoch_t_loss)

        # Validation Period
        model.eval()
        l = []
        loss_value = 0

        with torch.no_grad():
            for i, batch in enumerate(valid_dataloader):
                As, S, Ns = batch["A"], batch["S"], batch["NS"]
                pred = model.forward(S[:,3:6], As)
                if REAL_OUTPUT == True:
                    output_d = torch.hstack((Ns[:, :2], Ns[:, 5:]))
                else:
                    output_d = Ns
                loss = model.loss(pred, output_d)
                loss_value = loss.item()
                l.append(loss_value)
            epoch_v_loss = np.mean(l)
            validation_L.append(epoch_v_loss)

            if epoch_v_loss < min_loss:
                print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(epoch_t_loss))
                print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(epoch_v_loss))
                min_loss = epoch_v_loss

                PATH = log_path + '/best_model.pt'
                torch.save(model.state_dict(), PATH)
                abort_learning = 0
                decay_lr = 0
            else:
                abort_learning += 1
                decay_lr += 1
            scheduler.step(epoch_v_loss)

        np.savetxt(log_path + "training_L.csv", np.asarray(training_L))
        np.savetxt(log_path + "testing_L.csv", np.asarray(validation_L))

        if abort_learning > 20:
            break
        t1 = time.time()
        print(epoch, "time used: ", (t1 - t0) / (epoch + 1), "lr:", lr)

    return training_L, validation_L


def train_in_sim_offlinedata(env, parameters, train_batches, valid_batches, log_path, epochs, lr=1e-4, save=False,
                             plot=False):
    test_after_x_epochs = 500
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_L, validation_L = [], []
    min_loss = + np.inf

    rewards_curve = []
    abort_learning = 0
    decay_lr = 0
    t0 = time.time()
    for epoch in range(epochs):
        # Training Period
        l = []
        model.train()
        for data in train_batches:

            # time11 = time.time()
            IMGs, As, Ns = data
            IMGs, As, Ns = torch.from_numpy(IMGs).to(device), torch.from_numpy(As).to(device), torch.from_numpy(Ns).to(
                device)
            if BlACK_ACTION == True:
                As = torch.zeros_like(As)

            pred = model.forward(IMGs, As)
            if len(pred[0]) == 2:
                output_d = Ns[:, :2]
            else:
                output_d = Ns[:, :6]
            loss = model.loss(pred, output_d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l.append(loss.item())
            # time12 = time.time()
        #     print(time12-time11)
        # print('allbatchtime:',time.time()-time01)

        epoch_t_loss = np.mean(l)
        training_L.append(epoch_t_loss)

        # Validation Period
        model.eval()
        l = []
        with torch.no_grad():
            for data in valid_batches:
                IMGs, As, Ns = data
                IMGs, As, Ns = torch.from_numpy(IMGs).to(device), torch.from_numpy(As).to(device), torch.from_numpy(
                    Ns).to(device)

                if BlACK_ACTION == True:
                    As = torch.zeros_like(As)
                pred = model.forward(IMGs, As)
                if len(pred[0]) == 2:
                    output_d = Ns[:, :2]
                else:
                    output_d = Ns[:, :6]
                loss = model.loss(pred, output_d)
                loss_value = loss.item()
                l.append(loss_value)
            epoch_v_loss = np.mean(l)
            validation_L.append(epoch_v_loss)

            if epoch_v_loss < min_loss:
                print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(epoch_t_loss))
                print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(epoch_v_loss))
                min_loss = epoch_v_loss
                if epoch > 1:
                    PATH = log_path + '/best_model.pt'
                    torch.save(model.state_dict(), PATH)
                abort_learning = 0
                decay_lr = 0
            else:
                abort_learning += 1
                decay_lr += 1

        np.savetxt(log_path + "training_L.csv", np.asarray(training_L))
        np.savetxt(log_path + "testing_L.csv", np.asarray(validation_L))

        if decay_lr > 25:
            lr *= 0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if abort_learning > 30:
            break

        t1 = time.time()
        print(epoch, "time used: ", (t1 - t0) / (epoch + 1), "lr:", lr)

    return training_L, validation_L


def start_train_model(env, num_data, batch_size=8, lr=1e-1, use_DataLoader=False, scale=True,
                      input_previous_action=False):
    # data_model = 'data_%dk'%data_num

    train_packages = int(num_data * 0.8)
    valid_packages = int(num_data * 0.2)
    if train_packages + valid_packages != num_data:
        print("Error: can't separating data")
        quit()

    if use_DataLoader == True:
        all_A = []
        all_NS = []
        all_Done = []
        all_S = []
        for i in range(train_packages):
            NS = np.loadtxt(dataset_path + "%d/NS.csv" % (i))[:, :6]
            S = np.loadtxt(dataset_path + "%d/S.csv" % (i))[:, :6]
            A = np.loadtxt(dataset_path + "%d/A.csv" % i)
            Done = np.loadtxt(dataset_path + "%d/DONE.csv" % i)
            all_S.append(S)
            all_NS.append(NS)
            all_A.append(A)
            all_Done.append(Done)
        all_A = np.asarray(all_A)
        all_NS = np.asarray(all_NS)
        all_S = np.asarray(all_S)
        all_Done = np.asarray(all_Done)

        all_S = normalize_output_data(all_S, scale=scale)
        all_S = np.clip(all_S, -1, 1)

        all_NS = normalize_output_data(all_NS, scale=scale)
        all_NS = np.clip(all_NS, -1, 1)

        all_A = torch.from_numpy(all_A).to(device, dtype=torch.float)
        all_S = torch.from_numpy(all_S).to(device, dtype=torch.float)
        all_NS = torch.from_numpy(all_NS).to(device, dtype=torch.float)
        all_Done = torch.from_numpy(all_Done).to(device, dtype=torch.float)
        train_info = [all_A,all_S, all_NS, all_Done]

        vall_A = []
        vall_S = []
        vall_NS = []
        v_all_Done = []
        for i in range(train_packages, num_data):
            vS = np.loadtxt(dataset_path + "%d/S.csv" % (i))[:, :6]
            vNS = np.loadtxt(dataset_path + "%d/NS.csv" % (i))[:, :6]
            vA = np.loadtxt(dataset_path + "%d/A.csv" % i)[:, :12]
            v_Done = np.loadtxt(dataset_path + "%d/DONE.csv" % i)
            vall_S.append(vS)
            vall_NS.append(vNS)
            vall_A.append(vA)
            v_all_Done.append(v_Done)

        vall_S = np.asarray(vall_S)
        vall_A = np.asarray(vall_A)
        vall_NS = np.asarray(vall_NS)
        vall_Done = np.asarray(v_all_Done)

        vall_S = normalize_output_data(vall_S, scale=scale)
        vall_NS = normalize_output_data(vall_NS, scale=scale)

        vall_A = torch.from_numpy(vall_A).to(device, dtype=torch.float)
        vall_S = torch.from_numpy(vall_S).to(device, dtype=torch.float)
        vall_NS = torch.from_numpy(vall_NS).to(device, dtype=torch.float)
        vall_Done = torch.from_numpy(vall_Done).to(device, dtype=torch.float)
        valid_info = [vall_A,vall_S, vall_NS, vall_Done]
        print("data loaded!")

        train_L, test_L = train_in_sim(env,
                                       sin_para,
                                       num_data,
                                       batch_size,
                                       train_info,
                                       valid_info,
                                       log_path,

                                       lr=lr,
                                       epochs=5000,
                                       save=True,
                                       plot=True,
                                       input_previous_action=input_previous_action)

    plt.plot(np.arange(len(train_L)), train_L, label='training')
    plt.plot(np.arange(len(test_L)), test_L, label='validation')
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig(log_path + "lc.png")
    plt.show()


def normalize_output_data(data, scale=True):
    if len(data.shape) == 3:
        if scale == True:
            for kk in range(6):
                data[:, :, kk] = np.subtract(data[:, :, kk], xx1[kk])
                data[:, :, kk] *= xx2[kk]
    else:
        if scale == True:
            for kk in range(6):
                data[:, kk] = np.subtract(data[:, kk], xx1[kk])
                data[:, kk] *= xx2[kk]
    return data


if __name__ == '__main__':
    robot_idx = 0
    name = 'V%03d_cam' % robot_idx
    print(name)

    # sin_para = np.loadtxt("traj_optim/dataset/V000_sin_para.csv")
    sin_para = np.loadtxt("CADandURDF/robot_repo/V000_cam/0.csv")
    # 1 Collect Data
    # 2 Train
    # 3 Test
    # 4 Dataset evaluation
    RUN_PROGRAM = 3
    RAND_FIRCTION = True
    RAND_T = True
    RAND_P = True
    print("PROGRAM:",RUN_PROGRAM)

    if RUN_PROGRAM == 2:
        mode_name = "mode54"
        pre_trained_model = False
        pre_trained_model_path = "C:/Users/yuhan/Desktop/visual_self-model/train/mode32/best_model.pt"
        dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_mix4/"
        num_data = 1000
        batch_size = 128
        NORM = True
        PRE_A = True
        BLACK_IMAGE = False
        BlACK_ACTION = False
        use_DataLoader = True
        REAL_OUTPUT = False
        ACTIVATED_F = "L"  # Tahn or LeakReLU
        BLUR_IMG = True
        if REAL_OUTPUT:
            num_output = 3
        else:
            num_output = 6

        # model3: ResNet
        from ResNet_RNN import *

        model = IMU_bl(num_classes = num_output, ACTIVATED_F = ACTIVATED_F).to(device)

        scale_coff = np.loadtxt("norm_dataset_V000_cam_n0.2_mix4.csv")
        xx1, xx2 = scale_coff[0], scale_coff[1]
        env = 0

        model_save_path = "train/"
        log_path = model_save_path + "%s/" % mode_name
        print(mode_name)

        try:
            os.mkdir(log_path)
        except OSError:
            pass

        start_train_model(env,
                          lr=1e-4,
                          scale=NORM,
                          num_data=num_data,
                          batch_size=batch_size,
                          input_previous_action=PRE_A,
                          use_DataLoader=use_DataLoader)


    elif RUN_PROGRAM == 3:
        p.connect(p.DIRECT)
        GROUND_list = ['rug_rand','grid','color_dots','grass_rand']
        TASK_list = ['f','r','l','b']
        for i in range(1):
            for j in range(4):
                idx_num = 54
                GROUND = GROUND_list[i]
                TASK = TASK_list[j]

                NORM = True
                PRE_A = True
                BLACK_IMAGE = False
                BlACK_ACTION = False
                REAL_OUTPUT = False
                OPERATE = False
                LOG_A = True
                ACTIVATED_F ="L" # Tahn or LeakReLU
                if REAL_OUTPUT == True:
                    num_output = 3
                else:
                    num_output = 6


                load_trained_model_path = "train/mode%d/best_model.pt" % idx_num
                scale_coff = np.loadtxt("norm_dataset_V000_cam_n0.2_mix4.csv")
                xx1, xx2 = scale_coff[0], scale_coff[1]
                from ResNet_RNN import *

                model = IMU_bl(num_classes=num_output, ACTIVATED_F=ACTIVATED_F).to(device)

                model.load_state_dict(torch.load(load_trained_model_path, map_location=torch.device(device)))
                model.to(device)

                env = OpticalEnv(name,
                                 robot_camera=True,
                                 camera_capture=False,
                                 urdf_path="../CADandURDF/robot_repo/%s/urdf/%s.urdf" % (name, name),
                                 ground_type=GROUND,
                                 rand_fiction = RAND_FIRCTION,
                                 rand_torque = RAND_T,
                                 rand_pos= RAND_P)

                env.sleep_time = 0
                env.data_collection = True
                test_log_path = "test/test%d_%s_%s/" % (idx_num, TASK, GROUND)

                try:
                    os.mkdir(test_log_path)
                    os.mkdir(test_log_path + "/data/")
                    os.mkdir(test_log_path + "/img/")
                except OSError:
                    pass

                test_model(model, env,
                           epoisde_times=10,
                           num_traj=50,
                           parameters = sin_para,
                           noise=0.2,
                           step_num=56,
                           save_flag=False,
                           input_pre_a=PRE_A)

                ####-----------------------------------####
                ###########################################
