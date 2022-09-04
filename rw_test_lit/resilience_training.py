import numpy as np
from ResNet_RNN import *
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
import cv2
import random
import torchvision
import time

device = 'cuda'

def apply_vo_model(data_folder, step_num = 56):
    device = 'cuda'
    VO_load_trained_model_path = "../train/vo1/best_model.pt"
    scale_coff = np.loadtxt("../norm_dataset_V000_cam_n0.2_mix4.csv")
    xx1, xx2 = scale_coff[0], scale_coff[1]

    vo_model = OV_Net(ACTIVATED_F='L', img_channel=7, num_classes=6, normalization=True).to(device)
    vo_model.load_state_dict(torch.load(VO_load_trained_model_path, map_location=torch.device(device)))
    vo_model.to(device)
    vo_model.eval()

    vo_results = []
    for i in range(step_num):
        image_data = np.zeros((7,128,128))
        for j in range(7):
            if (i*6+j) != step_num*6:
                img = plt.imread(data_folder+"img/%d.jpeg"%(i*6+j))
            else:
                img = plt.imread(data_folder+"img/%d.jpeg"%(i*6+j-1))
            image_data[j,:,:] = np.copy(img)

        ov_IMGs = np.copy(image_data)
        ov_IMGs = np.asarray(ov_IMGs) / 255

        ov_IMGs = torch.from_numpy(ov_IMGs.astype(np.float32)).to(device)

        ov_IMGs = ov_IMGs.unsqueeze(0)
        vo_result = vo_model.forward(ov_IMGs)
        vo_result = vo_result.cpu().detach().numpy()

        for kk in range(6):
            vo_result[:,kk] /= xx2[kk]
            vo_result[:,kk] = np.add(vo_result[:,kk], xx1[kk])
        vo_results.append(vo_result[0])
    np.savetxt(data_folder+'vo_result.csv',np.asarray(vo_results))


class ImgData2(Dataset):
    def __init__(self, epoisod_pack_id, info):

        self.start_idx = epoisod_pack_id[0]
        self.end_idx = epoisod_pack_id[1]

        self.all_A, self.all_NS, self.idx_list = info

    def __getitem__(self, idx):
        # time0 = time.time()
        lookfor_id = 0
        epoch_id = 0
        while 1:
            lookfor_id += self.idx_list[epoch_id]

            if lookfor_id > idx:
                lookfor_id -= self.idx_list[epoch_id]
                epoch_id += 1
                break


        pack_id = epoch_id + self.start_idx
        idx_data = idx - lookfor_id
        root_dir =  "rw_data/%d/img/" % pack_id


        img_pack = []

        for i in range(1, 6):
            img_path0 = root_dir + "%d.jpeg" % (idx_data* 6 + i)
            img = plt.imread(img_path0)
            img_pack.append(img)

        if idx_data == 0 or idx ==0:
            pre_action = torch.zeros(12).to(device, dtype=torch.float)
        else:
            pre_action = self.all_A[idx-1]

        IMG = np.asarray(img_pack)  # [:, :, :1]
        if len(IMG.shape) == 4:
            IMG = IMG.squeeze()

        IMG = self.transform_img(IMG)

        # # time1 = time.time()
        next_state = self.all_NS[idx]
        action = self.all_A[idx]


        action = torch.concat((pre_action, action))

        sample = {'image': IMG, 'A': action, "NS": next_state}
        # time3 = time.time()
        # print("loaddata2:",time3-time0)

        return sample

    def __len__(self):
        num_data = 0
        for i in range(len(self.idx_list)):
            num_data += self.idx_list[i]
        print(num_data)
        return num_data


    def transform_img(self, img):

        if BLUR_IMG == True:
            if random.randint(0, 1) == 1:
                # cv2.imshow('IMG_origin',np.hstack((img[0],img[1],img[2],img[3],img[4])))
                sig_r_xy = random.uniform(0.1, 5)
                win_r = 2 * random.randint(1, 20) + 1
                img = cv2.GaussianBlur(img, (win_r, win_r), sigmaX=sig_r_xy, sigmaY=sig_r_xy,
                                       borderType=cv2.BORDER_DEFAULT)
                # cv2.imshow('IMG_Blurring', np.hstack((img[0], img[1], img[2], img[3], img[4])))
                # cv2.waitKey(0)
            IMG = torch.from_numpy(img).to(device, dtype=torch.float)

            if random.randint(0, 1) == 1:
                T = torchvision.transforms.ColorJitter(brightness=[0.1, 10])
                IMG = T(IMG)

        else:
            IMG = torch.from_numpy(img).to(device, dtype=torch.float)

        return IMG

BLUR_IMG = True
def re_train():

    mode_name = 'model103_broken_recovery_blur(b2)'
    pre_trained_model_path = "../train/mode103/best_model.pt"
    train_num_data = 40
    test_num_data = 5
    batch_size = 2

    # BLUR_IMG = True
    scale_coff = np.loadtxt("../norm_dataset_V000_cam_n0.2_mix4.csv")
    model = ResNet50(ACTIVATED_F='L', img_channel=5, num_classes=6, input_pre_a=True, normalization=True).to(
        device)
    xx1, xx2 = scale_coff[0], scale_coff[1]

    model_save_path = "../train/"
    log_path = model_save_path + "%s/" % mode_name
    print(mode_name)

    try:
        os.mkdir(log_path)
    except OSError:
        pass

    all_A = []
    all_NS = []
    step_num_list = []
    for i in range(train_num_data):
        if i == 0:
            all_A = np.loadtxt("rw_data/%d/A.csv" % i)
            all_NS = np.loadtxt("rw_data/%d/vo_result.csv" % i)
            continue
        A = np.loadtxt("rw_data/%d/A.csv" % i)
        vo = np.loadtxt("rw_data/%d/vo_result.csv" % i)
        all_A = np.vstack((all_A, A[1:]))
        all_NS= np.vstack((all_NS,vo[1:]))
        step_num_list.append(len(vo)-1)
    all_A = np.asarray(all_A)
    all_NS = np.asarray(all_NS)
    for kk in range(6):
        all_NS[:, kk] = np.subtract(all_NS[:, kk], xx1[kk])
        all_NS[:, kk] *= xx2[kk]
    all_A = torch.from_numpy(all_A).to(device, dtype=torch.float)
    all_NS = torch.from_numpy(all_NS).to(device, dtype=torch.float)
    train_info = [all_A, all_NS, step_num_list]


    vali_all_A = []
    vali_all_NS = []
    vali_step_num_list = []
    for i in range(train_num_data,train_num_data+test_num_data):
        print(i)
        if i == train_num_data:
            vali_all_A = np.loadtxt("rw_data/%d/A.csv" % i)
            vali_all_NS = np.loadtxt("rw_data/%d/vo_result.csv" % i)
            continue
        v_A = np.loadtxt("rw_data/%d/A.csv" % i)
        v_vo = np.loadtxt("rw_data/%d/vo_result.csv" % i)
        vali_all_A = np.vstack((vali_all_A, v_A[1:]))
        vali_all_NS= np.vstack((vali_all_NS,v_vo[1:]))
        vali_step_num_list.append(len(v_A)-1)
    vali_all_A = np.asarray( vali_all_A)
    vali_all_NS = np.asarray(vali_all_NS)
    for kk in range(6):
        vali_all_NS[:, kk] = np.subtract(vali_all_NS[:, kk], xx1[kk])
        vali_all_NS[:, kk] *= xx2[kk]

    vali_all_A = torch.from_numpy(vali_all_A).to(device, dtype=torch.float)
    vali_all_NS = torch.from_numpy(vali_all_NS).to(device, dtype=torch.float)
    test_info = [vali_all_A, vali_all_NS, vali_step_num_list]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

    training_data = ImgData2(info=train_info,
                             epoisod_pack_id=[0,train_num_data])
    validation_data = ImgData2(info=test_info,
                               epoisod_pack_id=[train_num_data,train_num_data+test_num_data])

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=0)


    training_L, validation_L = [], []
    min_loss = + np.inf
    abort_learning = 0
    decay_lr = 0
    t0 = time.time()

    for epoch in range(1000):
        l = []
        model.train()
        for i, batch in enumerate(train_dataloader):
            IMGs, As, Ns = batch["image"], batch["A"], batch["NS"]
            pred = model.forward(IMGs, As)
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
                IMGs, As, Ns = batch["image"], batch["A"], batch["NS"]
                pred = model.forward(IMGs, As)

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
        print(epoch, "time used: ", (t1 - t0) / (epoch + 1))

    return training_L, validation_L







if __name__ == '__main__':
    # Apply VO model
    # for idd in range(22,45):
    #     print(idd)
    #     data_folder = 'rw_data/%d/'%idd
    #     data_don = np.loadtxt(data_folder+'A.csv')
    #     step_num = len(data_don)
    #     apply_vo_model(data_folder,step_num)

    re_train()