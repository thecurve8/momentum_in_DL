# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 23:56:26 2020

@author: Alexander
"""
import copy
import torch
from training_helpers import test, get_accuracy
from datetime import datetime 

def train_loop_storm(model, trainloader, testloader, k, w, c, criterion,
                     epochs_to_run, log_interval, cuda):

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    last_model = copy.deepcopy(model)
    G_squarred_accu = []
    a = []
    d = []
    first_pass = True
    
    
    for epoch in epochs_to_run:
        model.train()
        running_loss = 0
        
        for (data, target) in trainloader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            
            #Zero out the gradients of the models. 
            model.zero_grad()
            last_model.zero_grad()
    
            loss = criterion(model(data), target)
            loss_last = criterion(last_model(data), target)
    
            #dloss/dx for every Variable 
            loss.backward()
            running_loss += loss.item() * data.size(0)

            loss_last.backward()

            params = model.parameters()
            last_params = last_model.parameters()
            zipped_params = zip(params, last_params)
            
            for i, (p, last_p) in enumerate(zipped_params):

                if first_pass:                    
                    p_grad = torch.clone(p.grad.data).detach()
                    G_squarred_accu.append(pow(torch.norm(p_grad), 2))
                    learning_rate = k/(pow((w+G_squarred_accu[i]),1/3)) 
                    
                    a.append(c*pow(learning_rate,2))
                    d.append(p_grad)
                    first_pass = False
                else:
                    last_p_grad = torch.clone(last_p.grad.data).detach()
                    p_grad = torch.clone(p.grad.data).detach()
                    G_squarred_accu[i] += pow(torch.norm(p_grad), 2)
                    learning_rate = k/(pow((w+G_squarred_accu[i]),1/3)) 
                    d[i] = p_grad + (1-a[i])*(d[i]-last_p_grad)
                    a[i] = c*pow(learning_rate,2)
                    last_model = copy.deepcopy(model)
                p.data -= learning_rate * d[i]

    
        
        train_loss_epoch = running_loss / len(trainloader.dataset)
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
            
    return train_losses, test_losses, train_accuracies, test_accuracies