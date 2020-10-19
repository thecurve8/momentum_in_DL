# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:05:26 2020

@author: Alexander
"""

import torch
from training_helpers import test, get_accuracy, train
from datetime import datetime 

def train_loop_optimizer(model, trainloader, testloader, optimizer, criterion, epochs_to_run, log_interval, cuda):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs_to_run):
        train_loss_epoch, model, optimizer = train(model, trainloader, optimizer, criterion, cuda)
        train_losses.append(train_loss_epoch)

        with torch.no_grad():
            test_loss_epoch = test(model, testloader, criterion, cuda)
            test_losses.append(test_loss_epoch)
        
        if epoch%log_interval == 0:
            train_acc = get_accuracy(model, trainloader, cuda)
            train_accuracies.append(train_acc)
            test_acc = get_accuracy(model, testloader, cuda)
            test_accuracies.append(test_acc)
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss_epoch:.4f}\t'
                  f'Test loss: {test_loss_epoch:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Test accuracy: {100 * test_acc:.2f}')
            
    return train_losses, test_losses, train_accuracies, test_accuracies, model, optimizer