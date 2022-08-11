mode_name = "mode1"
num_data = 200
batch_size = 128
NORM = True
PRE_A = False
BLACK_IMAGE = True
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
self.act_f = nn.Tanh()

mode_name = "mode2"
num_data = 400
batch_size = 128
NORM = True
PRE_A = True
BLACK_IMAGE = False
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
self.act_f = nn.Tanh()

mode_name = "mode3"
num_data = 400
batch_size = 128
NORM = True
PRE_A = False
BLACK_IMAGE = False
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
self.act_f = nn.Tanh()



mode_name = "mode4"
num_data = 400
batch_size = 128
NORM = True
PRE_A = True
BLACK_IMAGE = True
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F ="T" # Tahn or LeakReLU

mode_name = "mode5"
num_data = 600
batch_size = 8
NORM = True
PRE_A = True
BLACK_IMAGE = False
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F ="T" # Tahn or LeakReLU


mode_name = "mode6"
num_data = 400
batch_size = 128
NORM = True
PRE_A = True
BLACK_IMAGE = False
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F ="L" # Tahn or LeakReLU
mode 2 vs mode 6, mode 6 win


mode_name = "mode7"
num_data = 600
batch_size = 8
NORM = True
PRE_A = True
BLACK_IMAGE = False
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F = "L"  # Tahn or LeakReLU


mode_name = "mode8"
num_data = 600
batch_size = 128
NORM = True
PRE_A = True
BLACK_IMAGE = True
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F = "L"  # Tahn or LeakReLU


mode_name = "mode9"
num_data = 600
batch_size = 32
NORM = True
PRE_A = False
BLACK_IMAGE = False
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F = "L"  # Tahn or LeakReLU


mode_name = "mode10"
num_data = 600
batch_size = 128
NORM = True
PRE_A = False
BLACK_IMAGE = True
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F = "L"  # Tahn or LeakReLU


mode_name = "mode11"
num_data = 600
batch_size = 128
NORM = True
PRE_A = True
BLACK_IMAGE = False
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F = "L"  # Tahn or LeakReLU
BLUR_IMG = True


mode_name = "mode12"
num_data = 600
batch_size = 32
NORM = True
PRE_A = False
BLACK_IMAGE = False
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F = "L"  # Tahn or LeakReLU
BLUR_IMG = True

----
random torque and friction last 400 are random

mode_name = "mode13"
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

mode_name = "mode14"
num_data = 1000
batch_size = 128
NORM = True
PRE_A = True
BLACK_IMAGE = True
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F = "L"  # Tahn or LeakReLU
BLUR_IMG = True


mix data
mode_name = "mode15"
num_data = 100
batch_size = 32
NORM = True
PRE_A = True
BLACK_IMAGE = False
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F = "L"  # Tahn or LeakReLU
BLUR_IMG = True


mode_name = "mode16"
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


mode_name = "mode17"
pre_trained_model = True
pre_trained_model_path = "train/mode13/best_model.pt"
dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_mix/"
mode_name = "mode17"
num_data = 200
batch_size = 32
NORM = True
PRE_A = True
BLACK_IMAGE = False
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F = "L"  # Tahn or LeakReLU
BLUR_IMG = True




torque noise, action noise

mode_name = "mode18"

pre_trained_model = True
pre_trained_model_path = "train/mode17/best_model.pt"
dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_mix2/"

num_data = 200
batch_size = 32
NORM = True
PRE_A = True
BLACK_IMAGE = False
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F = "L"  # Tahn or LeakReLU
BLUR_IMG = True

mode_name = "mode19"
pre_trained_model = True
pre_trained_model_path = "train/mode17/best_model.pt"
dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_mix2/"
num_data = 400
batch_size = 32
NORM = True
PRE_A = True
BLACK_IMAGE = False
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F = "L"  # Tahn or LeakReLU
BLUR_IMG = True


Baselines1:
Prediction based on Action

Friction and toque random = False
mode_name = "mode20"
pre_trained_model = False
pre_trained_model_path = "train/mode17/best_model.pt"
dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_mix2/"
num_data = 100
batch_size = 128
NORM = True
PRE_A = True
BLACK_IMAGE = True
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F = "L"  # Tahn or LeakReLU


mode_name = "mode21"
pre_trained_model = True
pre_trained_model_path = "train/mode18/best_model.pt"
dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_rug_rand/"
num_data = 1000
batch_size = 8
NORM = True
PRE_A = True
BLACK_IMAGE = False
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F = "L"  # Tahn or LeakReLU
BLUR_IMG = True


mode_name = "mode23"
pre_trained_model = True
pre_trained_model_path = "C:/Users/yuhan/Desktop/visual_self-model/train/mode22/best_model.pt"
dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_rug_rand/"
num_data = 1000
batch_size = 32
NORM = True
PRE_A = True
BLACK_IMAGE = False
BlACK_ACTION = False
use_DataLoader = True
REAL_OUTPUT = False
ACTIVATED_F = "L"  # Tahn or LeakReLU
BLUR_IMG = True


        mode_name = "mode24"
        pre_trained_model = True
        pre_trained_model_path = "C:/Users/yuhan/Desktop/visual_self-model/train/mode23/best_model.pt"
        dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_mix2/"
        num_data = 400
        batch_size = 128
        NORM = True
        PRE_A = True
        BLACK_IMAGE = False
        BlACK_ACTION = False
        use_DataLoader = True
        REAL_OUTPUT = False
        ACTIVATED_F = "L"  # Tahn or LeakReLU
        BLUR_IMG = True

        mode_name = "mode25"
        pre_trained_model = True
        pre_trained_model_path = "C:/Users/yuhan/Desktop/visual_self-model/train/mode23/best_model.pt"
        dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_mix2/"
        num_data = 1000
        batch_size = 8
        NORM = True
        PRE_A = True
        BLACK_IMAGE = False
        BlACK_ACTION = False
        use_DataLoader = True
        REAL_OUTPUT = False
        ACTIVATED_F = "L"  # Tahn or LeakReLU
        BLUR_IMG = True
        scale_coff = np.loadtxt("norm_dataset_V000_cam_n0.2_rug_rand.csv")

        mode_name = "mode26"
        pre_trained_model = True
        pre_trained_model_path = "C:/Users/yuhan/Desktop/visual_self-model/train/mode25/best_model.pt"
        dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_mix3/"
        num_data = 400
        batch_size = 128
        NORM = True
        PRE_A = True
        BLACK_IMAGE = False
        BlACK_ACTION = False
        use_DataLoader = True
        REAL_OUTPUT = False
        ACTIVATED_F = "L"  # Tahn or LeakReLU
        BLUR_IMG = True
        scale_coff = np.loadtxt("norm_dataset_V000_cam_n0.2_mix3.csv")

mode name 30 series -> baselines


mix3 datatset
mode name 40 series -> baselines
CONSTRAIN == True

        mode_name = "mode41"
        pre_trained_model = False
        pre_trained_model_path = "C:/Users/yuhan/Desktop/visual_self-model/train/mode40/best_model.pt"
        dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_mix3/"
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
        scale_coff = np.loadtxt("norm_dataset_V000_cam_n0.2_mix3.csv")

        mode_name = "mode42"
        pre_trained_model = True
        pre_trained_model_path = "C:/Users/yuhan/Desktop/visual_self-model/train/mode41/best_model.pt"
        dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_mix3/"
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
        scale_coff = np.loadtxt("norm_dataset_V000_cam_n0.2_mix3.csv")


50 series: new dataset mix4

        mode_name = "mode50"
        pre_trained_model = False
        pre_trained_model_path = "C:/Users/yuhan/Desktop/visual_self-model/train/mode25/best_model.pt"
        dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_mix4/"
        num_data = 400
        batch_size = 128
        NORM = True
        PRE_A = True
        BLACK_IMAGE = False
        BlACK_ACTION = False
        use_DataLoader = True
        REAL_OUTPUT = False
        ACTIVATED_F = "L"  # Tahn or LeakReLU
        BLUR_IMG = True
        scale_coff = np.loadtxt("norm_dataset_V000_cam_n0.2_mix4.csv")


        mode_name = "mode51"
        pre_trained_model = True
        pre_trained_model_path = "C:/Users/yuhan/Desktop/visual_self-model/train/mode23/best_model.pt"
        dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_mix4/"
        num_data = 400
        batch_size = 128
        NORM = True
        PRE_A = True
        BLACK_IMAGE = False
        BlACK_ACTION = False
        use_DataLoader = True
        REAL_OUTPUT = False
        ACTIVATED_F = "L"  # Tahn or LeakReLU
        BLUR_IMG = True
        scale_coff = np.loadtxt("norm_dataset_V000_cam_n0.2_mix4.csv")


    elif RUN_PROGRAM == 2:
        mode_name = "mode52"
        pre_trained_model = False
        pre_trained_model_path = "C:/Users/yuhan/Desktop/visual_self-model/train/mode23/best_model.pt"
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
        scale_coff = np.loadtxt("norm_dataset_V000_cam_n0.2_mix4.csv")


mode_name = "mode53"
        pre_trained_model = False
        pre_trained_model_path = "C:/Users/yuhan/Desktop/visual_self-model/train/mode32/best_model.pt"
        dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_mix4/"
        num_data = 200
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

## model = IMU_bl(num_classes = num_output, ACTIVATED_F = ACTIVATED_F).to(device)

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

## model = IMU_bl(num_classes = num_output, ACTIVATED_F = ACTIVATED_F).to(device)


mode_name = "mode55"
        pre_trained_model = False
        pre_trained_model_path = "C:/Users/yuhan/Desktop/visual_self-model/train/mode23/best_model.pt"
        dataset_path = "C:/visual_project_data/data_package1/V000_cam_n0.2_mix4/"
        num_data = 1000
        batch_size = 128
        NORM = True
        PRE_A = True
##BLACK_IMAGE = True
        BlACK_ACTION = False
        use_DataLoader = True
        REAL_OUTPUT = False
        ACTIVATED_F = "L"  # Tahn or LeakReLU
##BLUR_IMG = False
        scale_coff = np.loadtxt("norm_dataset_V000_cam_n0.2_mix4.csv")

        if REAL_OUTPUT:
            num_output = 3
        else:
            num_output = 6

        # model3: ResNet
        from ResNet_RNN import *

        model = ResNet50( ACTIVATED_F,img_channel=5, num_classes=num_output, input_pre_a=PRE_A, normalization=NORM).to(device)



mode_name = "mode56"
        pre_trained_model = False
        pre_trained_model_path = "C:/Users/yuhan/Desktop/visual_self-model/train/mode23/best_model.pt"
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
    !!    BLUR_IMG = False
        scale_coff = np.loadtxt("norm_dataset_V000_cam_n0.2_mix4.csv")

        if REAL_OUTPUT:
            num_output = 3
        else:
            num_output = 6

        # model3: ResNet
        from ResNet_RNN import *

        # model = ResNet50( ACTIVATED_F,img_channel=5, num_classes=num_output, input_pre_a=PRE_A, normalization=NORM).to(device)
        # model = ResNet50( ACTIVATED_F,img_channel=5, num_classes=num_output, input_pre_a=PRE_A, normalization=NORM).to(device)
##model = ResNet152( ACTIVATED_F,img_channel=5, num_classes=num_output, input_pre_a=PRE_A, normalization=NORM).to(device)


