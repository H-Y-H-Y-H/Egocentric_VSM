import numpy as np
import random

inner_motor_index = [0, 3, 6, 9]
middle_motor_index = [1, 4, 7, 10]
outer_motor_index = [2, 5, 8, 11]

# def reset(i,flag = 0):
#     if flag ==1:
#         if i < 3:
#             action = np.array([0, -0.8, 0.9] * 4)
#         elif i < 10 and i >= 3:
#             action = np.array([0, 0, 0] * 4)
#         elif i < 20 and i >= 10:
#             action = np.array([0, 0, 0.9] * 4)
#         else:
#             action = np.array([0, -0.8, 0.9] * 4)
#     else:
#         action = np.array([-1, -0.8, 0.9] * 4)
#     return action

def rand_para_gss(para_batch):
    return np.random.normal(para_batch,0.1)



def random_para():
    para = np.zeros(16)
    for i in range(16):
        para[i] = random.uniform(-1, 1)
    for i in range(2,6):
        para[i] *= 2*np.pi
    return para


def batch_random_para(para_batch,Gaussian = False):
    if Gaussian == False:
        for i in range(16):
            para_batch[i][i] = random.uniform(-1, 1)
            if i in [2,3,4,5]:
                para_batch[i][i]*= 2*np.pi
    else:
        for i in range(16):
            para_batch[i][i] = np.random.normal(para_batch[i][i],scale = 0.1)
    return para_batch


def sin_move(ti, para, debug = False):
    # print(para)
    s_action = np.zeros(12)

    if debug == True:
        para[0] = 0
        para[1] = 0

    # print(ti)
    s_action[0] = para[0] * np.sin(ti / 8 * 2 * np.pi + para[2]) + para[10]  # left   hind
    s_action[3] = para[1] * np.sin(ti / 8 * 2 * np.pi + para[3]) + para[11]  # left   front
    s_action[6] = para[1] * np.sin(ti / 8 * 2 * np.pi + para[4]) - para[11]  # right  front
    s_action[9] = para[0] * np.sin(ti / 8 * 2 * np.pi + para[5]) - para[10]  # right  hind

    s_action[1] = para[6] * np.sin(ti / 8 * 2 * np.pi + para[2]) + para[12]  # left   hind
    s_action[4] = para[7] * np.sin(ti / 8 * 2 * np.pi + para[3]) + para[13]  # left   front
    s_action[7] = para[7] * np.sin(ti / 8 * 2 * np.pi + para[4]) + para[13]  # right  front
    s_action[10]= para[6] * np.sin(ti / 8 * 2 * np.pi + para[5]) + para[12]  # right  hind

    s_action[2] = para[8] * np.sin(ti / 8 * 2 * np.pi + para[2]) + para[14]  # left   hind
    s_action[5] = para[9] * np.sin(ti / 8 * 2 * np.pi + para[3]) + para[15]  # left   front
    s_action[8] = para[9] * np.sin(ti / 8 * 2 * np.pi + para[4]) + para[15]  # right  front
    s_action[11]= para[8] * np.sin(ti / 8 * 2 * np.pi + para[5]) + para[14]  # right  hind



    return s_action



def change_parameters(para):

    for i in range(16):
        rdm_number = random.uniform(-1, 1)
        if random.getrandbits(1):
            if i in [0,1,6,7,8,9]:
                para[i] = rdm_number
            elif i in range(2,6):
                para[i] = 2 * np.pi * rdm_number
            elif i in range(10,12):
                para[i] = rdm_number * (1 - abs(para[i - 10]))
            elif i in range(12, 16):
                para[i] = rdm_number * (1 - abs(para[i - 6]))
    return para

if __name__ == '__main__':
    # para_batch = np.array([random_para()]*16)
    # batch_random_para(para_batch)
    para_batch = [[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]]
    para_batch = np.asarray(para_batch)
    # para_batch[:2] = np.random.normal(para_batch[:2],0.1)
    for i in range(3):
        para_batch[i][i] = np.random.normal(para_batch[i][i],0.1)
    print(para_batch)
