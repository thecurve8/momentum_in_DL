# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:12:46 2020

@author: Alexander
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    A class used to model a LeNet-5 network
    http://yann.lecun.com/exdb/lenet/ and 
    https://towardsdatascience.com/convolutional-neural-network-champions-part-1-lenet-5-7a8d6eb98df6
    
    
    Methods
    -------
    forward(x)
        returns log_softmax output of the network with input x
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