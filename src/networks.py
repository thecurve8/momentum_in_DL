# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:12:46 2020

@author: Alexander


This file contains the different networks used in the experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    A class used to model a LeNet-5 network
    http://yann.lecun.com/exdb/lenet/ and 
    https://towardsdatascience.com/convolutional-neural-network-champions-part-1-lenet-5-7a8d6eb98df6

    """
    
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
      
    def forward(self, x):
        x=self.conv1(x)
        x=torch.tanh(x)
        x=F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x=self.conv2(x)
        x=torch.tanh(x)
        x=F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x=x.view(-1, 16*5*5)
        x=self.fc1(x)
        x=torch.tanh(x)
        x=self.fc2(x)
        x=torch.tanh(x)
        x=self.fc3(x)
        return F.log_softmax(x, dim=1)

class LeNet_batch_norm(nn.Module):
    """
    A class used to model a LeNet-5 network
    http://yann.lecun.com/exdb/lenet/ and 
    https://towardsdatascience.com/convolutional-neural-network-champions-part-1-lenet-5-7a8d6eb98df6
    with batch normalization used after conv layers
    
    Methods
    -------
    forward(x)
        returns log_softmax output of the network with input x
    """
    
    def __init__(self):
        super(LeNet_batch_norm, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.batch_norm2 = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.batch_norm3 = nn.BatchNorm1d(num_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.batch_norm4 = nn.BatchNorm1d(num_features=84)
        self.fc3 = nn.Linear(84, 10)
        self.batch_norm5 = nn.BatchNorm1d(num_features=10)
      
    def forward(self, x):
        x=self.conv1(x)
        x=self.batch_norm1(x)
        x=torch.tanh(x)
        x=F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x=self.conv2(x)
        x=self.batch_norm2(x)
        x=torch.tanh(x)
        x=F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x=x.view(-1, 16*5*5)
        x=self.fc1(x)
        x=self.batch_norm3(x)
        x=torch.tanh(x)
        x=self.fc2(x)
        x=self.batch_norm4(x)
        x=torch.tanh(x)
        x=self.fc3(x)
        x=self.batch_norm5(x)
        return F.log_softmax(x, dim=1)
    
class LeNet_layer_norm(nn.Module):
    """
    A class used to model a LeNet-5 network
    http://yann.lecun.com/exdb/lenet/ and 
    https://towardsdatascience.com/convolutional-neural-network-champions-part-1-lenet-5-7a8d6eb98df6
    with batch normalization used after conv layers
    
    Methods
    -------
    forward(x)
        returns log_softmax output of the network with input x
    """
    
    def __init__(self):
        super(LeNet_layer_norm, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2, stride=1)
        self.layer_norm1 = nn.LayerNorm([6,28,28])
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.layer_norm2 = nn.LayerNorm([16,10,10])

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.layer_norm3 = nn.LayerNorm(120)
        self.fc2 = nn.Linear(120, 84)
        self.layer_norm4 = nn.LayerNorm(84)
        self.fc3 = nn.Linear(84, 10)
        self.layer_norm5 = nn.LayerNorm(10)
      
    def forward(self, x):
        x=self.conv1(x)
        x=self.layer_norm1(x)
        x=torch.tanh(x)
        x=F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x=self.conv2(x)
        x=self.layer_norm2(x)
        x=torch.tanh(x)
        x=F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x=x.view(-1, 16*5*5)
        x=self.fc1(x)
        x=self.layer_norm3(x)
        x=torch.tanh(x)
        x=self.fc2(x)
        x=self.layer_norm4(x)
        x=torch.tanh(x)
        x=self.fc3(x)
        x=self.layer_norm5(x)
        return F.log_softmax(x, dim=1)
    
###############################################################################
#########---------------------RESNET-----------------------------------########
###############################################################################
### code adapted from : https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=True):
        super(BasicBlock, self).__init__()
        self.use_batchnorm = use_batchnorm
        
        self.conv1 = conv3x3(in_planes, planes, stride)
        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        if self.use_batchnorm:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if self.use_batchnorm:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
                )

    def forward(self, x):
        if self.use_batchnorm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
            out += self.shortcut(x)
            out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_batchnorm = True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.use_batchnorm = use_batchnorm
        
        self.conv1 = conv3x3(3,64)
        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_batchnorm=self.use_batchnorm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_batchnorm=self.use_batchnorm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_batchnorm=self.use_batchnorm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, use_batchnorm=self.use_batchnorm)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, use_batchnorm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_batchnorm:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10, use_batchnorm=True):
    """
    Return ResNet18 model.

    Parameters
    ----------
    num_classes : int, optional
        Number of prediction classes. The default is 10.
    use_batchnorm : TYPE, optional
        Whether to use batchnorm layers in the model. The default is True.

    Returns
    -------
    torch.nn.Module
        ResNet18 model.

    """
    return ResNet(BasicBlock, [2,2,2,2], num_classes, use_batchnorm)

def ResNet34(num_classes=10, use_batchnorm=True):
    """
    Return ResNet34 model.

    Parameters
    ----------
    num_classes : int, optional
        Number of prediction classes. The default is 10.
    use_batchnorm : TYPE, optional
        Whether to use batchnorm layers in the model. The default is True.

    Returns
    -------
    torch.nn.Module
        ResNet34 model.

    """
    return ResNet(BasicBlock, [3,4,6,3], num_classes, use_batchnorm)

def ResNet50(num_classes=10, use_batchnorm=True):
    """
    Return ResNet50 model.

    Parameters
    ----------
    num_classes : int, optional
        Number of prediction classes. The default is 10.
    use_batchnorm : TYPE, optional
        Whether to use batchnorm layers in the model. The default is True.

    Returns
    -------
    torch.nn.Module
        ResNet50 model.

    """
    return ResNet(BasicBlock, [3,4,6,3], num_classes, use_batchnorm)

def ResNet101(num_classes=10, use_batchnorm=True):
    """
    Return ResNet101 model.

    Parameters
    ----------
    num_classes : int, optional
        Number of prediction classes. The default is 10.
    use_batchnorm : TYPE, optional
        Whether to use batchnorm layers in the model. The default is True.

    Returns
    -------
    torch.nn.Module
        ResNet101 model.

    """
    return ResNet(BasicBlock, [3,4,23,3], num_classes, use_batchnorm)

def ResNet152(num_classes=10, use_batchnorm=True):
    """
    Return ResNet152 model.

    Parameters
    ----------
    num_classes : int, optional
        Number of prediction classes. The default is 10.
    use_batchnorm : TYPE, optional
        Whether to use batchnorm layers in the model. The default is True.

    Returns
    -------
    torch.nn.Module
        ResNet152 model.

    """
    return ResNet(BasicBlock, [3,8,36,3], num_classes, use_batchnorm)