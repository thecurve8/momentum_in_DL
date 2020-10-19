# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 19:56:22 2020

@author: Alexander
"""

import torch

def test(model, testloader, criterion, cuda):
    model.eval()
    running_loss = 0
    for data, target in testloader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)

    test_loss = running_loss / len(testloader.dataset)
    return test_loss
    
def train(model, trainloader, optimizer, criterion, cuda):
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

def get_accuracy(model, dataloader, cuda):
    '''
    Function for computing the accuracy of the predictions over the entire dataloader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for data, target in dataloader:
            if cuda:
                data, target = data.cuda(), target.cuda()   

            predictions = model(data)
            _, predicted_labels = torch.max(predictions, 1)

            n += target.size(0)
            correct_pred += (predicted_labels == target).sum()

    return float(correct_pred) / n