import numpy as np
import matplotlib.pyplot as plt

# data_path = "data_package/V500_noise0.05/0/NS.csv"
label = ["dx","dy","dz","r","p","y"]
mode = "V000_cam_n0.2_mix0810"
_path ="C:/visual_project_data/data_package1/%s"%mode

def norm_scale():
    tem_save = []
    tem_save2 = []
    for j in range(200):
        data_path = _path +"/%d/NS.csv"%j
        data = np.loadtxt(data_path)[:,:6]
        all_mean = np.mean(data,axis=0)

        data = np.subtract(data,all_mean)
        ed_n = np.linalg.norm(data,axis=0)
        sc = 10/ed_n
        tem_save.append(sc)
        tem_save2.append(all_mean)
    scale_num1 = np.mean(tem_save,axis=0)
    scale_num2 = np.mean(tem_save2, axis=0)

    return scale_num2,scale_num1

def plot_data(scale = False):

    data_path = _path +"/%d/NS.csv"%100
    data = np.loadtxt(data_path)
    for i in range(10):
        data_path = _path +"/%d/NS.csv"%i
        data = np.vstack((data, np.loadtxt(data_path)))
    for i in range(6):
        if scale == True:
            target = np.subtract(data[:, i], scale_num1[i])
            target = target * scale_num2[i]
            # target /= scale_num2[i]
            # target = np.add(data[:, i], scale_num1[i])
            # target += 1
        else:
            target = data[:, i]

        # target /= scale_num2[i]
        # target = np.add(data[:, i], scale_num1[i])
        plt.plot(range(len(target)),target,label=label[i],alpha = 0.7)
        print(np.mean(target))
        print(np.std(target))
        print(np.min(target))
        print(np.max(target))
        print("--------")

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

scale_num1,scale_num2 = norm_scale()
scale_num2 = np.asarray(scale_num2)
scale_num1 = np.asarray(scale_num1)
a = np.asarray([scale_num1,scale_num2])
np.savetxt("norm_dataset_%s.csv"%mode,a)
norm = np.loadtxt("norm_dataset_%s.csv"%mode)
# norm = np.loadtxt("norm_dataset_V000_cam_n0.2_rug_rand.csv")

# scale_num1, scale_num2 = norm[0],norm[1]
# print(scale_num1,scale_num2)
plot_data(scale = True)



