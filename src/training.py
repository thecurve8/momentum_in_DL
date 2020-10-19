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

def main_train_loop(algo, model, trainloader, testloader, *args):
    available_algo_names = ('SVRG', 'ADAM', 'SGD')
    if not isinstance(algo, str):
        raise TypeError("Expected str for algo. Got {}".format(type(algo)))
    if algo not in available_algo_names:
        raise ValueError("Expected algo value in "+ str(available_algo_names) +
                          " got {}".format(algo))
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Expected torch.nn.Module for model. Got: {}".format(type(model)))
    if not isinstance(args['epochs'], int) :
        raise TypeError("Expected int for epochs. Got {}".format(type(args['epochs'])))
    if args['epochs'] < 1:
        raise ValueError("Expected strictly positive for epochs. Got {}".format(args['epochs']))
    if not (isinstance(args['lr'], float) or isinstance(args['lr'], int)):
        raise TypeError("Expected float or int for lr. Got {}".format(type(args['lr'])))
    if args['lr']<=0:
        raise ValueError("Expected strictly positive value for lr. Got {}".format(args['lr']))
    if not (isinstance(args['momentum'], float) or isinstance(args['momentum'], int)):
        raise TypeError("Expected float or int for momentum. Got {}".format(type(args['momentum'])))
    if args['momentum']<0:
        raise ValueError("Expected non-negative value for momentum. Got {}".format(args['momentum']))
    if not (isinstance(args['svrg_freq'], float), isinstance(args['svrg_freq'], int)):
        raise TypeError("Expected float or int for svrg_freq. Got {}".format(type(args['svrg_freq'])))
    if args['svrg_freq'] <= 0:
        raise ValueError("Expected strictly positive value for svrg_freq. Got {}".format(args['svrg_freq']))
    
    batch_size = trainloader.batch_size
    
    if batch_size != args['batch_size']:
        raise Warning("batch_size argument and trainloader batch_size are different. {} - {}".format(
            args['batch_size'], batch_size))
    
    test_batch_size = testloader.batch_size
    if test_batch_size != args['test_batch_size']:
        raise Warning("test_batch_size argument and testloader batch_size are different. {} - {}".format(
            args['test_batch_size'], test_batch_size))
    
    if args['cuda']:
        model.cuda()
    
    criterion = nn.CrossEntropyLoss()

    if algo == 'SGD' or algo == 'ADAM':
        if algo == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args['lr'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=args['lr'])

        train_losses, test_losses, train_accuracies, test_accuracies, model, \
        optimizer = train_loop_optimizer(model, trainloader, testloader,
                            optimizer, criterion, args['epochs'],
                            log_interval=args['log_interval'], cuda=args['cuda'])
    else:
        train_losses, test_losses, train_accuracies, test_accuracies, model, \
        snapshot_model, curr_batch_iter = \
            train_loop_SVRG(model, trainloader, testloader, args['lr'], 
                            freq = args['svrg_freq']*len(trainloader.dataset)/batch_size, 
                            criterion = criterion, epochs_to_run=args['epochs'],
                            log_interval=args['log_interval'], cuda=args['cuda'])
