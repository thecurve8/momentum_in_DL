# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:14:02 2020

@author: Alexander

Contains the main methods used to run the experiments with all algorithms.
"""

import torch 
import torch.optim as optim
from svrg import train_loop_SVRG
from train_optimizer import train_loop_optimizer
from storm_optim import train_loop_storm_optim
from torch.optim.lr_scheduler import MultiStepLR

def create_arguments(batch_size=32, test_batch_size=32, epochs=20,
                    lr=0.001, momentum=0.0, svrg_freq=5.0, seed=1,
                    log_interval = 1, cuda=True, k=0.1, w=0.1, c=0.1):
    """
    Create basic arguments for model training

    Parameters
    ----------
    batch_size : int, optional
        The default is 32.
    test_batch_size : int, optional
        The default is 32.
    epochs : int, optional
        Epochs to run. The default is 20.
    lr : float, optional
        Learning rate. The default is 0.001.
    momentum : float, optional
        Momentum to use when applicanle. The default is 0.0.
    svrg_freq : float, optional
        Frequency after which to update mu in SVRG. Counted in whole epochs.
        The default is 5.0.
    seed : int, optional
        The default is 1.
    log_interval : int, optional
        Number of epochs netween logs. The default is 1.
    cuda : bool, optional
        cuda available. The default is True.
    k : float, optional
        k parameter for STORM. The default is 0.1 (from paper)
    w : float, optional
        w parameter for STORM. The default is 0.1 (from paper)
    c : float, optional
        c parameter for STORM. The default is 0.1 (arbitarily)

    Returns
    -------
    args : dict
        Dictionary of parameters.

    """
    
    args={}
    args['batch_size']=batch_size
    args['test_batch_size']=test_batch_size
    args['epochs']=epochs
    args['lr']=lr
    args['momentum']=momentum
    args['svrg_freq']=svrg_freq
    args['seed']=seed
    args['log_interval']=log_interval
    args['cuda'] = cuda
    args['k'] = k
    args['w'] = w
    args['c'] = c
    return args

def build_return_dict_optim(train_losses, test_losses, train_accuracies,
                            test_accuracies, model_state_dict):
    """
    Creates a dictionary with the output of the train_loop with optimizer

    Parameters
    ----------
    train_losses : list of float
        train losses after each epoch.
    test_losses : list of float
        test losses after each epoch..
    train_accuracies : list of float
        train accuracies after each epoch.
    test_accuracies : list of float
        test accuracies after each epoch.
    model_state_dict : dict
        trained model state dictionary.

    Returns
    -------
    return_values : dict
        dictionary with mapped values.

    """
    return_values = {}
    return_values['train_losses']=train_losses
    return_values['test_losses']=test_losses
    return_values['train_accuracies']=train_accuracies
    return_values['test_accuracies']=test_accuracies
    return_values['model_state_dict']=model_state_dict
    return return_values

def build_return_dict_svrg(train_losses, test_losses, train_accuracies,
                           test_accuracies, model_state_dict, snapshot_model_state_dict,
                           curr_batch_iter):
    """
    Creates a dictionary with the output of the train_loop with svrg
   

    Parameters
    ----------
    train_losses : list of float
        train losses after each epoch.
    test_losses : list of float
        test losses after each epoch..
    train_accuracies : list of float
        train accuracies after each epoch.
    test_accuracies : list of float
        test accuracies after each epoch.
    model_state_dict : dict
        State dictionary of the trained model.
    snapshot_model_state_dict : dict
        State dictionary of the snapchot model.
    curr_batch_iter : int
        batch iteration after training.

    Returns
    -------
    return_values : dict
        dictionary with mapped values.
    """
    
    return_values = {}
    return_values['train_losses']=train_losses
    return_values['test_losses']=test_losses
    return_values['train_accuracies']=train_accuracies
    return_values['test_accuracies']=test_accuracies
    return_values['model_state_dict']=model_state_dict
    return_values['snapshot_model_state_dict']=snapshot_model_state_dict
    return_values['curr_batch_iter']=curr_batch_iter
    return return_values

def build_return_dict_storm(train_losses, test_losses, train_accuracies,
                           test_accuracies, model_state_dict):
    """
    Creates a dictionary with the output of the train_loop with svrg
   

    Parameters
    ----------
    train_losses : list of float
        train losses after each epoch.
    test_losses : list of float
        test losses after each epoch..
    train_accuracies : list of float
        train accuracies after each epoch.
    test_accuracies : list of float
        test accuracies after each epoch.
    model_state_dict : dict
        dictionary of the state of the model after training

    Returns
    -------
    return_values : dict
        dictionary with mapped values.
    """
    
    return_values = {}
    return_values['train_losses']=train_losses
    return_values['test_losses']=test_losses
    return_values['train_accuracies']=train_accuracies
    return_values['test_accuracies']=test_accuracies
    return_values['model_state_dict']=model_state_dict

    return return_values

def train_loop(algo, model, trainloader, testloader, criterion, args, milestones=None, gamma=None):
    """
    Main train loop for all algorithms

    Parameters
    ----------
    algo : str
        'SVRG', 'ADAM','SGD', 'STORM' or 'ADAGRAD.
    model : torch.nn.Module
        model to train
    trainloader : torch.utils.data.DataLoader
        Trainloader object.
    testloader : torch.utils.data.DataLoader
        Testloader object.
    criterion : torch.nn._Loss
        loss function.
    args : dict
        Dictionary with basic arguments used for training.
    milestones : list of int
        milestones to schedule optimizers
    gamma : float
        rate at which to decrease lr

    Raises
    ------
    TypeError
        Check type of every given argument.
    ValueError
        Check value of every given argument.
    Warning
        Raises warning if the specified batch_size or test_batch_size doesn't 
        match the trainloader or testloader.


    Returns
    -------
    dict
        dictionary with output of the training loop.

    """
    available_algo_names = ('SVRG', 'ADAM', 'SGD', 'STORM', 'ADAGRAD')
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
        
    if (algo == 'SGD') or (algo == 'ADAM') or (algo=='ADAGRAD'):
        if algo == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
        elif algo == 'ADAM':
            optimizer = optim.Adam(model.parameters(), lr=args['lr'])
        elif algo == 'ADAGRAD':
            optimizer = optim.Adagrad(model.parameters(), lr=args['lr'])
        scheduler = None
        if milestones and gamma:
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        train_losses, test_losses, train_accuracies, test_accuracies,\
        model_state_dict = train_loop_optimizer(model, trainloader, testloader,
                            optimizer, criterion, args['epochs'],
                            log_interval=args['log_interval'], cuda=args['cuda'],
                            scheduler=scheduler)
        
        return build_return_dict_optim(train_losses, test_losses,
                                       train_accuracies, test_accuracies, model_state_dict)
    
    elif algo == 'SVRG':
        train_losses, test_losses, train_accuracies, test_accuracies, model_state_dict, \
        snapshot_model_state_dict, curr_batch_iter = \
            train_loop_SVRG(model, trainloader, testloader, args['lr'], 
                            freq = args['svrg_freq']*len(trainloader), 
                            criterion = criterion, epochs_to_run=args['epochs'],
                            log_interval=args['log_interval'], cuda=args['cuda'],
                            milestones=milestones, gamma=gamma)
            
        return build_return_dict_svrg(train_losses, test_losses, train_accuracies,
                                      test_accuracies, model_state_dict,
                                      snapshot_model_state_dict, curr_batch_iter )
    elif algo == 'STORM' : 
        train_losses, test_losses, train_accuracies, test_accuracies, model_state_dict = \
            train_loop_storm_optim(model, trainloader, testloader, k=args['k'],
                                   w=args['w'], c=args['c'], criterion = criterion,
                                   epochs_to_run=args['epochs'], 
                                   log_interval=args['log_interval'], cuda=args['cuda'])

        return build_return_dict_storm(train_losses, test_losses,
                                       train_accuracies, test_accuracies, model_state_dict)
