# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:14:02 2020

@author: Alexander
"""

import torch 
import torch.nn as nn
import torch.optim as optim
from svrg import train_loop_SVRG
from train_optimizer import train_loop_optimizer

def main_train_loop(algo, model, epochs_to_run, lr, momentum, svrg_freq,
                    trainloader, testloader, log_interval, cuda):
    available_algo_names = ('SVRG', 'ADAM', 'SGD')
    if not isinstance(algo, str):
        raise TypeError("Expected str for algo. Got {}".format(type(algo)))
    if algo not in available_algo_names:
        raise ValueError("Expected algo value in "+ str(available_algo_names) +
                          " got {}".format(algo))
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Expected torch.nn.Module for model. Got: {}".format(type(model)))
    if not isinstance(epochs_to_run, int) :
        raise TypeError("Expected int for epochs_to_run. Got {}".format(type(epochs_to_run)))
    if epochs_to_run < 1:
        raise ValueError("Expected strictly positive for epochs_to_run. Got {}".format(epochs_to_run))
    if not (isinstance(lr, float) or isinstance(lr, int)):
        raise TypeError("Expected float or int for lr. Got {}".format(type(lr)))
    if lr<=0:
        raise ValueError("Expected strictly positive value for lr. Got {}".format(lr))
    if not (isinstance(momentum, float) or isinstance(momentum, int)):
        raise TypeError("Expected float or int for momentum. Got {}".format(type(momentum)))
    if momentum<0:
        raise ValueError("Expected non-negative value for momentum. Got {}".format(momentum))
    if not (isinstance(svrg_freq, float), isinstance(svrg_freq, int)):
        raise TypeError("Expected float or int for svrg_freq. Got {}".format(type(svrg_freq)))
    if svrg_freq <= 0:
        raise ValueError("Expected strictly positive value for svrg_freq. Got {}".format(svrg_freq))
    
    batch_size = trainloader.batch_size
    
    if cuda:
        model.cuda()
    
    criterion = nn.CrossEntropyLoss()

    if algo == 'SGD' or algo == 'ADAM':
        if algo == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses, test_losses, train_accuracies, test_accuracies, model, \
        optimizer = train_loop_optimizer(model, trainloader, testloader,
                            optimizer, criterion, epochs_to_run,
                            log_interval=log_interval, cuda=cuda)
    else:
        train_losses, test_losses, train_accuracies, test_accuracies, model, \
        snapshot_model, curr_batch_iter = \
            train_loop_SVRG(model, trainloader, testloader, lr, 
                            freq = svrg_freq*len(trainloader.dataset)/batch_size, 
                            criterion = criterion, epochs_to_run=epochs_to_run,
                            log_interval=log_interval, cuda=cuda)
