import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

folder_name = '../res_test/'


def plot_curve(pred,gt,vo_result,id_name):
    vo_pred_loss_list = []
    for i in range(len(pred[0])):
        plt.figure(figsize = (4,4))
        plt.plot(range(len(gt)),gt[:,i],label = 'gt',linewidth = 2, alpha=0.3,c = 'r',)
        # plt.plot(range(len(gt)),pred[:,i],label = 'pred',c = 'g')
        plt.plot(range(len(gt)-1),vo_result[1:,i],label = 'vo',c = 'b')
        # min_y = np.min(gt[:,i])
        # max_y = np.max(min_y/2)
        # plt.vlines(done_list, ymin =min_y, ymax=max_y,label = "done")

        # plt.ylabel('loss')
        # plt.xlabel("step")
        plt.legend()
        plt.title("Evaluation_%s:"%d_path+label_list[i])
        plt.show()
        # plt.savefig("../plots/%s_%s"%(label_list[i],id_name))
        R_value, _ = pearsonr(gt[:, i], pred[:, i])
        loss= np.linalg.norm(gt[:, i]-pred[:,i])
        print("VSM:",R_value, loss)
        R_value, _ = pearsonr(gt[:-1, i], vo_result[1:, i])
        loss= np.linalg.norm(gt[:-1, i]-vo_result[1:, i])
        print("VO:", R_value, loss)
        vo_pred_loss_list.append(np.linalg.norm(vo_result[1:, i]-pred[:-1,i]))
    print(vo_pred_loss_list,np.sum(vo_pred_loss_list))


tasks_list = ['f', 'b', 'r', 'l']
ground_list = ["rug_rand", "grid", "color_dots", 'grass_rand']
plot_ground_list = ["Rug", "Grid", "Color dots", 'Grass']
label_list = ['dx', 'dy', 'dz', 'roll', 'pitch', 'yaw']
x_labels = ['Move\nForward', 'Move\nBackward', 'Turn\nRight', 'Turn\nLeft']

USE_MEDIAN_AND_RANGE = False
# mode_list = [0, 20, 13, 52]
# mode_list = [0, 20, 33, 52] # sin, action-only, IMU input

mode_name = ["sin", "action-only", "IMU only", "ours"]
mode_list = [101] # sin, action-only, IMU input


# fig = plt.figure(figsize=(8, 4))
st_peroid = 0
end_peroid =1
for ground_id in range(1):
    for mode in mode_list:
        loss_mean = []
        loss_std = []
        rsquare_mean = []
        rsquare_std = []
        score_mean = []
        score_std = []
        for task_id in range(1):
            task = tasks_list[task_id]
            if mode == 0:
                d_path = folder_name + "/baselines"
            else:
                d_path = folder_name + "/test%d_%s_%s" % (mode, task, ground_list[ground_id])

            gt = np.loadtxt(d_path + "/data/gt.csv")[56*st_peroid:56*end_peroid]
            vo_result = np.loadtxt(d_path + "/data/vo_result.csv")[56*st_peroid:56*end_peroid]
            pred = np.loadtxt(d_path + "/data/pred.csv")[56*st_peroid:56*end_peroid]
            robot_loc_ori = np.loadtxt(d_path + "/data/rob_pos_ori.csv")
            id_name = "%d_%s_%s.png"%(mode,task,ground_list[ground_id])
            plot_curve(pred, gt,vo_result,id_name)


# def change_plot():




