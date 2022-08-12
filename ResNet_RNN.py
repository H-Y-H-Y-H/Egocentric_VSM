import torch
import torch.nn as nn


class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=(1,1)):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=(1,1), stride=(1,1), padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=(3,3),
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):

        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes,input_pre_a = False, normalization = False,ACTIVATED_F = "T"):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.norm = normalization
        if ACTIVATED_F == "L":
            self.act_f = nn.LeakyReLU()
        else:
            self.act_f = nn.Tanh()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.input_pre_a = input_pre_a
        self.rnn = nn.LSTM(input_size =5, hidden_size=256, num_layers=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc0 = nn.Linear(512 * 4, 256 * 4)
        self.fc1 = nn.Linear(256 * 4, 512)
        self.fc2 = nn.Linear(512, 256)


        if self.input_pre_a == True:
            self.fc_a0 = nn.Linear(24, 64)
        else:
            self.fc_a0 = nn.Linear(12, 64)
        self.fc_a1 = nn.Linear(64, 128)
        self.fc_a2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, num_classes)


    def forward(self, x_IMG, x_Action):

        if self.norm == True:
            x_IMG = x_IMG*2-1
        x = self.conv1(x_IMG)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.act_f(self.fc0(x))
        x = self.act_f(self.fc1(x))
        x = self.act_f(self.fc2(x))

        x = torch.cat(5*[x.unsqueeze(0)])


        l_x = torch.transpose(x, 0, 2)
        x, _ = self.rnn(l_x)
        x = x[-1]


        x_a = self.act_f(self.fc_a0(x_Action))
        x_a = self.act_f(self.fc_a1(x_a))
        x_a = self.act_f(self.fc_a2(x_a))

        if len(x_a.shape) == 4:
            x_a = torch.squeeze(x_a)
        x = torch.concat((x,x_a),dim=1)
        x = self.act_f(self.fc3(x))
        x = self.fc4(x)
        x = self.fc5(x)

        return x

    def loss(self, pred, target):
        return torch.mean((pred - target) ** 2)

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)



class IMU_bl(nn.Module):
    def __init__(self, num_classes, ACTIVATED_F = "T"):
        super(IMU_bl, self).__init__()
        if ACTIVATED_F == "L":
            self.act_f = nn.LeakyReLU()
        else:
            self.act_f = nn.Tanh()


        self.fc0 = nn.Linear(3, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)

        self.fc_a0 = nn.Linear(24, 64)
        self.fc_a1 = nn.Linear(64, 128)
        self.fc_a2 = nn.Linear(128, 256)

        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, num_classes)


    def forward(self, x_State, x_Action):
        x = self.act_f(self.fc0(x_State))
        x = self.act_f(self.fc1(x))
        x = self.act_f(self.fc2(x))

        x_a = self.act_f(self.fc_a0(x_Action))
        x_a = self.act_f(self.fc_a1(x_a))
        x_a = self.act_f(self.fc_a2(x_a))

        x = torch.concat((x,x_a),dim=1)
        x = self.act_f(self.fc3(x))
        x = self.fc4(x)

        return x

    def loss(self, pred, target):
        return torch.mean((pred - target) ** 2)


class ov_model(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes, normalization = False,ACTIVATED_F = "T"):
        super(ov_model, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.norm = normalization
        if ACTIVATED_F == "L":
            self.act_f = nn.LeakyReLU()
        else:
            self.act_f = nn.Tanh()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.rnn = nn.LSTM(input_size =7, hidden_size=256, num_layers=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc0 = nn.Linear(512 * 4, 256 * 4)
        self.fc1 = nn.Linear(256 * 4, 512)
        self.fc2 = nn.Linear(512, 256)


        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, num_classes)


    def forward(self, x_IMG):

        if self.norm == True:
            x_IMG = x_IMG*2-1
        x = self.conv1(x_IMG)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.act_f(self.fc0(x))
        x = self.act_f(self.fc1(x))
        x = self.act_f(self.fc2(x))

        x = torch.cat(7*[x.unsqueeze(0)])


        l_x = torch.transpose(x, 0, 2)
        x, _ = self.rnn(l_x)
        x = x[-1]

        x = self.act_f(self.fc3(x))
        x = self.fc4(x)
        x = self.fc5(x)
        return x


    def loss(self, pred, target):
        # print(pred.get_device(),target.get_device())
        # if target.get_device() == -1:
        #     print(target)
        target = target.to('cuda', dtype=torch.float)
        return torch.mean((pred - target) ** 2)

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

def ResNet18(ACTIVATED_F,img_channel=3, num_classes=6, input_pre_a = False,normalization = True):
    return ResNet(block, [2, 2, 2, 2], img_channel, num_classes,input_pre_a = input_pre_a,normalization = normalization,ACTIVATED_F = ACTIVATED_F)

def ResNet50(ACTIVATED_F,img_channel=3, num_classes=6, input_pre_a = False,normalization = True):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes,input_pre_a = input_pre_a,normalization = normalization,ACTIVATED_F = ACTIVATED_F)

def OV_Net(ACTIVATED_F,img_channel=3, num_classes=6, normalization = True):
    return ov_model(block, [3, 4, 6, 3], img_channel, num_classes,normalization = normalization,ACTIVATED_F = ACTIVATED_F)


def ResNet101(ACTIVATED_F,img_channel=3, num_classes=6, input_pre_a = False,normalization = True):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes,input_pre_a = input_pre_a,normalization = normalization,ACTIVATED_F = ACTIVATED_F)


def ResNet152(ACTIVATED_F,img_channel=3, num_classes=6, input_pre_a = False,normalization = True):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes,input_pre_a = input_pre_a,normalization = normalization,ACTIVATED_F = ACTIVATED_F)


def test():
    net = ResNet101("L",img_channel=3, num_classes=1000)
    y = net(torch.randn(4, 3, 224, 224)).to("cuda")
    print(y.size())


if __name__ == "__main__":
    import time
    from torchsummary import summary
    if torch.cuda.is_available():device = 'cuda'
    else: device = 'cpu'

    # Plot summary and test speed
    print("start", device)
    model = ResNet18(ACTIVATED_F= 'L', img_channel=5, num_classes=6, input_pre_a = True).to(device)

    x_i = torch.randn(5, 128, 128).to(device)
    x_a = torch.randn(24).to(device)
    # summary(model, [(5, 128, 128),(1,1,24)])
    t1= time.time()
    t0 = t1
    for i in range(100):
        x_i = torch.randn(50, 5, 128, 128).to(device)
        x_a = torch.randn(50, 24).to(device)
        x_i = torch.clip(x_i,0,1)
        model.forward(x_i,x_a)
        t2 = time.time()
        print(t2-t1)
        t1 = time.time()

    t3 = time.time()
    print("all",(t3-t0)/100)

