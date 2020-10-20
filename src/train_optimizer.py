# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:05:26 2020

@author: Alexander

"""

import torch
from training_helpers import test, get_accuracy
from datetime import datetime 

def train_epoch_optimizer(model, trainloader, optimizer, criterion, cuda):
    """
    Train the model for one step with an optimizer

    Parameters
    ----------
    model : torch.nn.Module
        model to train
    trainloader : torch.utils.data.DataLoader
        Trainloader object.
    optimizer : torch.nn.Optimizer
        optimizer to use.
    criterion : torch.nn._Loss
        loss function.
    cuda : bool
        cuda available.

    Returns
    -------
    loss_epoch : float
        train loss of the epoch (average).
    model : torch.nn.Module
        trained model.
    optimizer : torch.nn.Optimizer
        used optimizer.

    """
    
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        #Zero out the gradients of the model. 
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)

        #dloss/dx for every Variable 
        loss.backward()
        #to do a one-step update on our parameter.
        optimizer.step()

        running_loss += loss.item() * data.size(0)
    loss_epoch = running_loss / len(trainloader.dataset)
    return loss_epoch, model, optimizer

def train_loop_optimizer(model, trainloader, testloader, optimizer, criterion, epochs_to_run, log_interval, cuda):
    """
    Training loop for a model with a given optimizer and criterion

    Parameters
    ----------
    model : torch.nn.Module
        model to train
    trainloader : torch.utils.data.DataLoader
        Trainloader object.
    testloader : torch.utils.data.DataLoader
        Testloader object.
    optimizer : torch.nn.Optimizer
        optimizer to use.
    criterion : torch.nn._Loss
        loss function.
    epochs_to_run : int
        epochs to train the model.
    log_interval : int
        number of epochs between logs.
    cuda : bool
        cuda available.

    Returns
    -------
    train_losses : list of float
        train losses after each epoch.
    test_losses : list of float
        test losses after each epoch..
    train_accuracies : list of float
        train accuracies after each epoch.
    test_accuracies : list of float
        test accuracies after each epoch.
    model : torch.nn.Module
        trained model.
    optimizer : torch.nn.Optimizer
        used optimizer.

    """
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs_to_run):
        train_loss_epoch, model, optimizer = train_epoch_optimizer(model, trainloader, optimizer, criterion, cuda)
        train_losses.append(train_loss_epoch)

        with torch.no_grad():
            test_loss_epoch = test(model, testloader, criterion, cuda)
            test_losses.append(test_loss_epoch)
            train_acc = get_accuracy(model, trainloader, cuda)
            train_accuracies.append(train_acc)
            test_acc = get_accuracy(model, testloader, cuda)
            test_accuracies.append(test_acc)
            
        if epoch%log_interval == 0: 
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss_epoch:.4f}\t'
                  f'Test loss: {test_loss_epoch:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Test accuracy: {100 * test_acc:.2f}')
    model_state_dict = {k: v.cpu() for k, v in model.state_dict()}

    return train_losses, test_losses, train_accuracies, test_accuracies, model_state_dict