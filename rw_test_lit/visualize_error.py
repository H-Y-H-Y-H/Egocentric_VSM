import numpy as np
from env import *
from camera_pybullet import *
from CADandURDF.robot_repo.control import *
import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
from ResNet_RNN import *
from model import *
from scipy.stats import pearsonr



def apply_vo_model():
    device = 'cuda'
    VO_load_trained_model_path = "../train/vo1/best_model.pt"
    scale_coff = np.loadtxt("../norm_dataset_V000_cam_n0.2_mix4.csv")
    xx1, xx2 = scale_coff[0], scale_coff[1]

    vo_model = OV_Net(ACTIVATED_F='L', img_channel=7, num_classes=6, normalization=True).to(device)
    vo_model.load_state_dict(torch.load(VO_load_trained_model_path, map_location=torch.device(device)))
    vo_model.to(device)
    vo_model.eval()

    vo_results = []
    for i in range(56):
        image_data = np.zeros((7,128,128))
        for j in range(7):
            if (i*6+j) != 336:
                img = plt.imread(data_folder+"img/%d.jpeg"%(i*6+j))
            else:
                img = plt.imread(data_folder + "img/335.jpeg" )
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



def plot_curve(vo,vsm,n_vo,n_vsm):
    loss_list = []
    std_list = []

    n_loss_list = []
    n_std_list = []
    for i in range(len(vo[0])):
        plt.figure(figsize=(8, 4))
        # plt.plot(vsm[:-1, i], label='pred', alpha=0.3, c='g')
        # # plt.plot(abnorm[:, i], label='abnorm', alpha=0.3, c='b')
        #
        # plt.plot(vo[1:, i], label='vo', linewidth=2, c='r', )
        #
        # plt.ylabel('value')
        # plt.xlabel("step")
        # plt.legend()
        # plt.show()
        # plt.savefig("../plots/%s_%s"%(label_list[i],id_name))
        R_value, _ = pearsonr(vo[1:, i], vsm[:-1, i])
        loss = np.mean((vo[1:, i]-vsm[:-1, i])**2)
        std = np.std((vo[1:, i]-vsm[:-1, i])**2)
        print("VSM:", R_value, loss)
        loss_list.append(loss)
        std_list.append(std)

        n_loss = np.mean((n_vo[1:, i]-n_vsm[:-1, i])**2)
        n_std = np.std((n_vo[1:, i]-n_vsm[:-1, i])**2)

        n_loss_list.append(n_loss)
        n_std_list.append(n_std)


    print(np.sum(loss_list))
    print(np.std(loss_list))

    print(np.sum(n_loss_list))
    print(np.std(n_loss_list))

if __name__ == '__main__':

    n_data_folder = "V000_103_normal2/"
    # data_folder = "V000_103_(res50_frozen)/"
    # data_folder = "V000_103_broken_0.051/"
    data_folder = "V000_103_(res100_broken)_0.036/"

    # Apply VO_model
    # apply_vo_model()

    # Visualize data:
    n_vo = np.loadtxt(n_data_folder + 'vo_result.csv')
    n_vsm = np.loadtxt(n_data_folder + 'pred.csv')

    vo = np.loadtxt(data_folder + 'vo_result.csv')
    vsm = np.loadtxt(data_folder + 'pred.csv')

    plot_curve(vo,vsm,n_vo,n_vsm)



