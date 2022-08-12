import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import utils, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os



class RCNN(nn.Module):

    def __init__(self,in_channels):
        super(RCNN, self).__init__()

        self.act_f = nn.Tanh()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        # 3，16，180，128
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        # 16，32，96，64
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        # 32，32，96，64
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        # 32，64，48, 32

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        # 64，64，48, 32
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.fc1 = nn.Linear(8192, 4096)
        self.fc2 = nn.Linear(4096, 1024)


        self.rnn = nn.LSTM(input_size =7, hidden_size=256, num_layers=1)

        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(256 ,128)
        self.fc5 = nn.Linear(128 ,32)
        self.fc6 = nn.Linear(32 ,6)


        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        # in: [7，128，128], out: [16，240，160]
        out = self.layer1(x)

        # in: [16，240，160], out: [32，121，81]
        out = self.layer2(out)

        # in: [32，121，81], out: [32，121，81]
        out3 = self.layer3(out)

        # in: [32，121，81], out: [64，59，41]
        out4 = self.layer4(torch.add(out3, out))

        # in: [64，59，41], out: [128，30，21]
        out = self.layer5(out4)
        out = self.layer6(torch.add(out4, out))
        out = self.layer7(out)
        out = out.reshape(out.size(0), -1)

        out = F.relu(self.fc1(out), inplace=True)
        out = self.drop(out)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.drop(out)
        out = F.relu(self.fc3(out), inplace=True)
        out = self.drop(out)

        out = torch.cat(7*[out.unsqueeze(0)])
        l_x = torch.transpose(out, 0, 2)
        x, _ = self.rnn(l_x)
        x = x[-1]

        x = F.relu(self.fc4(x), inplace=True)
        x = self.fc5(x)
        x = self.fc6(x)
        return x

    def loss(self, pred, target):
        target = target.to('cuda', dtype=torch.float)
        return torch.mean((pred - target) ** 2)

if __name__ == "__main__":
    import time
    from torchsummary import summary
    if torch.cuda.is_available():device = 'cuda'
    else: device = 'cpu'

    # Plot summary and test speed
    print("start", device)
    model = RCNN(in_channels=7).to(device)

    x_i = torch.randn(7, 128, 128).to(device)
    x_a = torch.randn(24).to(device)
    # summary(model, [(5, 128, 128),(1,1,24)])
    t1= time.time()
    t0 = t1
    for i in range(100):
        x_i = torch.randn(50, 7, 128, 128).to(device)
        # x_a = torch.randn(50, 24).to(device)
        x_i = torch.clip(x_i,0,1)
        model.forward(x_i)
        t2 = time.time()
        print(t2-t1)
        t1 = time.time()

    t3 = time.time()
    print("all",(t3-t0)/100)

