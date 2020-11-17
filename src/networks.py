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
### code comes from : https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

def conv3x3(in_channels, out_channels, stride = 1, groups = 1, dilation = 1):
    """
    3x3 convolution with padding

    Parameters
    ----------
    in_channels : int
        channels before convolution.
    out_channels : int
        channels after convolution.
    stride : int, optional
        stride controls the stride for the cross-correlation. The default is 1.
    groups : int, optional
        groups controls the connections between inputs and outputs. in_channels 
        and out_channels must both be divisible by groups. For example,
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels, and producing half the output channels, and both subsequently concatenated.
        At groups= in_channels, each input channel is convolved with its own set of filters. 
        The default is 1.
    dilation : int, optional
        Dilation controls the spacing between the kernel points. The default is 1.

    Returns
    -------
    nn.Conv2d
        3x3 convolution with padding.

    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_channels, out_channels, stride = 1):
    """
    1x1 convolution

    Parameters
    ----------
    in_channels : int
        channels before convolution.
    out_channels : int
        channels after convolution.
    stride : int, optional
        stride controls the stride for the cross-correlation. The default is 1.

    Returns
    -------
    nn.Conv2d
        1x1 convolution.

    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        ## CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        ## END
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

    
def _resnet(block, layers, **kwargs):
    """
    

    Parameters
    ----------
    block : BasicBlock
        DESCRIPTION.
    layers : List[int]
        DESCRIPTION.
    **kwargs : Any
        DESCRIPTION.

    Returns
    -------
    ResNet
        DESCRIPTION.

    """
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs) -> ResNet:
    """
    

    Parameters
    ----------
    **kwargs : Any
        DESCRIPTION.

    Returns
    -------
    ResNet
        DESCRIPTION.

    """
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs) 