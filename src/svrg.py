# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:32:50 2020

@author: Alexander
"""
import copy
import torch
from training_helpers import test, get_accuracy
from datetime import datetime 


def calculate_mu(model_snapchot, dataloader, criterion, cuda):
    print("Calculating new mu")
    for data, target in dataloader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        criterion(model_snapchot(data), target).backward()
    mu = []
    for param in model_snapchot.parameters():
        if param.grad != None:
            param.grad.data.div_(len(dataloader))
            mu.append(param.grad.data.clone())
    print("New mu calculated")
    return mu

def update_grad_svrg(params, snapshot_params, mu):
    if not len(params) == len(snapshot_params) == len(mu):
        raise ValueError("Expected input of identical length. "
                    "Got {}, {}, {}".format(len(params),
                                            len(snapshot_params),
                                            len(mu)))

    for i in range(len(mu)):
        params[i].grad.data.sub_(snapshot_params[i].grad.data)
        params[i].grad.data.add_(mu[i])

    
def svrg_train_epoch(model, snapshot_model, trainloader, learning_rate, freq, criterion, start_batch_iter, mu, cuda):
    model.train()
    running_loss = 0
    curr_batch_iter = start_batch_iter
    for batch_idx, (data, target) in enumerate(trainloader):

        if curr_batch_iter%freq == 0:
            print("Creatig new model snapshot at batch iteration {}".format(curr_batch_iter))
            snapshot_model = copy.deepcopy(model)
            if cuda:
                snapshot_model = snapshot_model.cuda()
            mu = calculate_mu(snapshot_model, trainloader, criterion, cuda)

        if cuda:
            data, target = data.cuda(), target.cuda()

        #Zero out the gradients of the models. 
        model.zero_grad()
        snapshot_model.zero_grad()

        loss = criterion(model(data), target)
        loss_snap = criterion(snapshot_model(data), target)

        #dloss/dx for every Variable 
        loss.backward()
        loss_snap.backward()

        params = model.parameters()
        snapshot_params = snapshot_model.parameters()
        #update gradients of parameters to reflect the SVRG algo
        update_grad_svrg(list(params), list(snapshot_params), mu)

        for param in model.parameters():
            param.data -= learning_rate * param.grad.data

        running_loss += loss.item() * data.size(0)

        curr_batch_iter+=1

    loss_epoch = running_loss / len(trainloader.dataset)

    return loss_epoch, model, snapshot_model, curr_batch_iter, mu

def train_loop_SVRG(model, trainloader, testloader, learning_rate, freq,
                    criterion, epochs_to_run, log_interval, cuda,
                    milestones=None, gamma=None):
    """
    Training loop of SVRG

    Parameters
    ----------
    model : torch.nn.Module
        model to train
    trainloader : torch.utils.data.DataLoader
        Trainloader object.
    testloader : torch.utils.data.DataLoader
        Testloader object.
    learning_rate : float
        Learning rate to use.
    freq : int
        frequency of gradient estimate (in epochs).
    criterion : torch.nn._Loss
        loss function.
    epochs_to_run : int
        Number of epochs to train.
    llog_interval : int
        number of epochs between logs.
    cuda : bool
        cuda available.
    milestones : list of int, optional
        If set, List of epochs at which to change the learning rate. The default is None.
    gamma : float, optional
        If set, the constant with which to scale the learning rate at each milestone. The default is None.

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
    model_state_dict : torch.nn.Module
        trained model.
    snapshot_model_state_dict : torch.nn.Module
        snapshot model.
    curr_batch_iter : int
        Last batch iteration.

    """
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    curr_batch_iter = 0
    snapshot_model = copy.deepcopy(model)
    mu = None
    used_learning_rate = learning_rate
    
    for epoch in range(epochs_to_run):
        if milestones and gamma:
            if epoch in milestones:
                used_learning_rate *= gamma
        train_loss_epoch, model, snapshot_model, curr_batch_iter, mu = \
            svrg_train_epoch(model, snapshot_model, trainloader, used_learning_rate,
                       freq, criterion, curr_batch_iter, mu, cuda)
            
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
    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    snapshot_model_state_dict = {k: v.cpu() for k, v in snapshot_model.state_dict().items()}
    return train_losses, test_losses, train_accuracies, test_accuracies, model_state_dict, snapshot_model_state_dict, curr_batch_iter    