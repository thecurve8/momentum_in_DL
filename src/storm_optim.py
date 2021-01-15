# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:45:09 2020

@author: Alexander
"""

import torch
from training_helpers import test, get_accuracy
from datetime import datetime 


class STORM(torch.optim.Optimizer):
    '''
    Implements STOchastic Recursive Momentum (STORM)
    Args:
        k (float, optional): learning rate scaling (called k in the original paper).
        c (float, optional): 
        w (float, optional): initial value of denominator in adaptive learning rate
    '''
    def __init__(self, params, c=10, k=0.1, w=0.1):
        defaults = dict(k=k, c=c, w=w)
        super().__init__(params, defaults)
        self.step_num = 0
        self.total_time = 0
    
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    raise Exception('Call update_momentum() first.')

                state['step'] += 1
                dt, sum_Gt_sq = state['dt'], state['sum_Gt_sq']
                w, k, c = group['w'], group['k'], group['c']

                eta_t = k / torch.pow(w + sum_Gt_sq, 1/3)
                p.data.add_(-eta_t, dt)
                state['at'] = min(1, c * eta_t**2)

        self.step_num += 1

    def update_momentum(self):
        for group in self.param_groups:
            for p in group['params']:
                gt = p.grad.data
                if gt is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    assert self.step_num == 0
                    # State initialization
                    state['step'] = 0
                    state['sum_Gt_sq'] =torch.sum(gt*gt)
                    state['recorded_gt'] = torch.zeros_like(p.data)
                    state['dt'] = gt.clone()
                    state['at'] = 1
                    continue

                gt_prev = state['recorded_gt']
                #assert not torch.allclose(gt, gt_prev), 'Please call clone_grad() in ' 'the previous step. '
                state['sum_Gt_sq'] += torch.sum(gt*gt)
                dt = state['dt']
                state['dt'] = gt + (1-state['at'])*(dt - gt_prev)
                # destroy previous cloned gt for safety
                state['recorded_gt'] = None

    def clone_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                gt = p.grad.data
                if gt is None:
                    continue
                self.state[p]['recorded_gt'] = gt.clone().detach()


def train_loop_storm_optim(model, trainloader, testloader, k, w, c, criterion,
                     epochs_to_run, log_interval, cuda,
                     milestones = None, gamma = None):
    """
    

    Parameters
    ----------
    model : torch.nn.Module
        model to train.
    trainloader : torch.utils.data.DataLoader
        Trainloader object..
    testloader : torch.utils.data.DataLoader
        Testloader object..
    k : float
        learning rate scaling (called k in the original paper)..
    w : float
        STORM parameter.
    c : float
        initial value of denominator in adaptive learning rate.
    criterion : torch.nn._Loss
        Loss function.
    epochs_to_run : int
        Epochs for training.
    log_interval : int
        number of epochs between logs..
    cuda : bool
        cuda available.
    milestones : list of int, optional
        If set the epochs at which to reschedule the learning rate. The default is None.
    gamma : list of int, optional
        If set, the amout to rescale the learing rate. The default is None.

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
    model_state_dict : dict
        Final state of the model.

    """
    
    optmz = STORM(model.parameters(), c, k, w)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    first_pass = True
    for epoch in range(epochs_to_run):
        running_loss = 0
        for data, target in trainloader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            
            loss = criterion(model(data), target)
            optmz.zero_grad()
            loss.backward()
            
            if first_pass:
                optmz.update_momentum()
                first_pass = False
            else:
                # record the grad for updating momentum
                optmz.clone_grad()
                # put a zero_grad() here just to show that step() only depend on the momentum
                optmz.zero_grad()
                # update to new weights x_{t+1} using previous momemtum
                optmz.step()
                # forward pass using new weights x_{t+1}, calculate \nabla f(x_{t+1}, \xi_{t+1})
                loss = criterion(model(data), target)
                # optmz.zero_grad() # the gradients should be already zero
                loss.backward()
                # update the momentum d_t
                optmz.update_momentum()
            running_loss += loss.item() * data.size(0)
            
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
    
    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    return train_losses, test_losses, train_accuracies, test_accuracies, model_state_dict    
    