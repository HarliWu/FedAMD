# -*- coding: utf-8 -*-

'''
This is full set for cifar datasets (CIFAR-10 and CIFAR100) 
Models: CNN, LeNet, AlexNet, ResNet, LR, VGG
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()

#         self.covn1 = nn.Sequential(  # (1,28,28)
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
#             # if stride=1, padding=(kernal_size-1)/2=(5-1)/2=2, #(16,28,28)
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, ),  # (16,14,14)
#         )
#         self.covn2 = nn.Sequential(
#             nn.Conv2d(16, 32, 5, 1, 2),  # (32,14,14)
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # (32,7,7)
#         )
#         self.out = nn.Linear(32 * 7 * 7, 10)

#     def forward(self, x):
#         x = self.covn1(x)
#         x = self.covn2(x)  # (batch,32,7,7)
#         x = x.view(x.size(0), -1)  # (batch, 32*7*7)
#         output = self.out(x)
#         return output

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)), 
            nn.ReLU(), 
            nn.Dropout(0.25)
        )

        self.fc1 = nn.Linear(64*11*11, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=0.5)
        out = self.fc2(out)
        return out


class MLP(nn.Module):
    def __init__(self, in_dims=784, out_dims=10, dim_hidden=100):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dims, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, out_dims)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        
        return out
        

        
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    from torchinfo import summary
    # print("ResNet8")
    model = MLP()
    summary(model, (1, 1, 28, 28), depth=5, verbose=1)
    # print("\n\n\n")
    # model = ResNet8(0, False, 200)
    # torchsummary.summary(model, (3, 64, 64), depth=5, verbose=1)