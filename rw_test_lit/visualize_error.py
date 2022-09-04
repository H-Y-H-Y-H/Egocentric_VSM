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
from resilience_training import apply_vo_model




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
    data_folder = "recovery3_0.021/V000_103_broken_recovery_blur/"

    # Apply VO_model
    # apply_vo_model(data_folder,step_num=56)

    # Visualize data:
    n_vo = np.loadtxt(n_data_folder + 'vo_result.csv')
    n_vsm = np.loadtxt(n_data_folder + 'pred.csv')

    vo = np.loadtxt(data_folder + 'vo_result.csv')
    vsm = np.loadtxt(data_folder + 'pred.csv')

    plot_curve(vo,vsm,n_vo,n_vsm)


