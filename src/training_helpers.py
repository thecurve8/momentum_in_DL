# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 19:56:22 2020

@author: Alexander

Contains methods to train already implemented optimizers.
"""

import torch

def test(model, testloader, criterion, cuda):
    """
    Test the current model with a criterion.

    Parameters
    ----------
    model : torch.nn.Module
        trained model.
    testloader : torch.utils.data.DataLoader
        Testloader object.
    criterion : torch.nn._Loss
        loss function.
    cuda : bool
        cuda available..

    Returns
    -------
    test_loss : float
        loss of the whole test dataset in the testloader.

    """
    
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
    
def get_accuracy(model, dataloader, cuda):
    """
    Function for computing the accuracy of the predictions over the entire dataloader

    Parameters
    ----------
    model : torch.nn.Module
        model to use.
    dataloader : torch.utils.data.DataLoader
        data loader object.
    cuda : bool
        cuda available.

    Returns
    -------
    accuracy : float
        accuracy of the model for the given dataloader.

    """
    
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
    accuracy = float(correct_pred) / n
    return accuracy