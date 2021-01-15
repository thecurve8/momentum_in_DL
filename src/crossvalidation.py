# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:52:54 2020

@author: Alexander

Contains methods to perform cross-validation on the different optimization algorithms to select their hyperparameters.
"""
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from train_optimizer import train_loop_optimizer
from svrg import train_loop_SVRG
from storm_optim import train_loop_storm_optim
import copy

def build_k_indices(n, K, seed):
    """Build k indices for k-fold cross-validation.
    
    Parameters
    ----------
    n : int
        total number of indices
    K : int
        Number of folds
    seed : int
        Seed for index shuffling
    Returns
    -------
    res : numpy array
        2-dimensional array with shuffled indices arranged in K rows
    """
    interval = int(n / K)
    np.random.seed(seed)
    # Shuffle (1, ..., num_row)
    indices = np.random.permutation(n)
    # Arrange indices into K lists
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(K)]
    res = np.array(k_indices)
    return res

def create_train_val_dataloader(k, k_indices, dataset, args):
    """
    Creates Train and Validation dataloaders.

    Parameters
    ----------
    k : int
        number of the folds.
    k_indices : array
        2-dimensional array with shuffled indices arranged in k rows, corresponding to indices for each fold.
    dataset : dataset
        Dataset from which to create the dataloaders.
    args : dict
        Dictionary with the arguments of the experiment.

    Returns
    -------
    trainloader : torch.utils.data.DataLoader
        Trainloader object.
    validationloader : torch.utils.data.DataLoader
        Testloader object.

    """
    train_indices = np.delete(k_indices, k, args['seed']).flatten()
    validation_indices = k_indices[k]

    trainloader = DataLoader(Subset(dataset, train_indices),
                                              batch_size=args['batch_size'],
                                              shuffle=True)
    validationloader = DataLoader(Subset(dataset, validation_indices),
                                                   batch_size=args['batch_size'],
                                                   shuffle=True)
    return trainloader, validationloader

def cross_validation_adam(model_initial, learning_rates, dataset, K, criterion, args):
    """
    Performs cross-validation for learning rate of ADAM.

    Parameters
    ----------
    model_initial : torch.nn.Module
        model to train.
    learning_rates : list of float
        List of learning rates to try.
    dataset : dataset
        Dataset from which to do the cross-validation.
    K : int
        Number of folds.
    criterion :  torch.nn._Loss
        loss function.
    args : dict
        Dictionary with eperiment parameters.

    Returns
    -------
    return_dict : dict
        dict with results of cross-vallidation.

    """
    
    k_indices = build_k_indices(len(dataset), K, args['seed'])
    
    len_learning_rate = len(learning_rates)

    training_errors_during_training = np.zeros((len_learning_rate, K, args['epochs']))
    validation_errors_during_training = np.zeros((len_learning_rate, K, int(args['epochs']//args['log_interval']) ))
    
    training_accuracies_during_training = np.zeros((len_learning_rate, K, args['epochs']))
    validation_accuracies_during_training = np.zeros((len_learning_rate, K, int(args['epochs']//args['log_interval']) ))
    
    for i, lr in enumerate(learning_rates):
        print("Training for learning rate={}".format(lr))
        for j, k in enumerate(range(K)):
            print("Fold {}".format(k))
            model = copy.deepcopy(model_initial)
            trainloader, validationloader = create_train_val_dataloader(k, k_indices, dataset, args)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            train_losses, val_losses, train_accuracies, val_accuracies, _ =\
                train_loop_optimizer(model, trainloader, validationloader,
                                     optimizer, criterion, args['epochs'],
                                     args['log_interval'], args['cuda'])
            
            training_errors_during_training[i,j] = train_losses
            validation_errors_during_training[i,j] = val_losses
            training_accuracies_during_training[i,j] = train_accuracies
            validation_accuracies_during_training[i,j] = val_accuracies
    return_dict = build_return_dict_CV(training_errors_during_training,
                                       validation_errors_during_training,
                                       training_accuracies_during_training,
                                       validation_accuracies_during_training,
                                       "CV adam lr:{}-{}".format(learning_rates[0], learning_rates[-1]))        
    return return_dict

def cross_validation_adagrad(model_initial, learning_rates, dataset, K, criterion, args):
    """
    Performs cross-validation for learning rate of AdaGrad.

    Parameters
    ----------
    model_initial : torch.nn.Module
        model to train.
    learning_rates : list of float
        List of learning rates to try.
    dataset : dataset
        Dataset from which to do the cross-validation.
    K : int
        Number of folds.
    criterion :  torch.nn._Loss
        loss function.
    args : dict
        Dictionary with eperiment parameters.


    Returns
    -------
    return_dict : dict
        dict with results of cross-vallidation.

    """
    
    k_indices = build_k_indices(len(dataset), K, args['seed'])
    
    len_learning_rate = len(learning_rates)

    training_errors_during_training = np.zeros((len_learning_rate, K, args['epochs']))
    validation_errors_during_training = np.zeros((len_learning_rate, K, int(args['epochs']//args['log_interval']) ))
    
    training_accuracies_during_training = np.zeros((len_learning_rate, K, args['epochs']))
    validation_accuracies_during_training = np.zeros((len_learning_rate, K, int(args['epochs']//args['log_interval']) ))
    
    for i, lr in enumerate(learning_rates):
        print("Training for learning rate={}".format(lr))
        for j, k in enumerate(range(K)):
            print("Fold {}".format(k))
            model = copy.deepcopy(model_initial)
            trainloader, validationloader = create_train_val_dataloader(k, k_indices, dataset, args)
            optimizer = optim.Adagrad(model.parameters(), lr=lr)

            train_losses, val_losses, train_accuracies, val_accuracies, _ =\
                train_loop_optimizer(model, trainloader, validationloader,
                                     optimizer, criterion, args['epochs'],
                                     args['log_interval'], args['cuda'])
            
            training_errors_during_training[i,j] = train_losses
            validation_errors_during_training[i,j] = val_losses
            training_accuracies_during_training[i,j] = train_accuracies
            validation_accuracies_during_training[i,j] = val_accuracies
    return_dict = build_return_dict_CV(training_errors_during_training,
                                       validation_errors_during_training,
                                       training_accuracies_during_training,
                                       validation_accuracies_during_training,
                                       "CV adagrad lr:{}-{}".format(learning_rates[0], learning_rates[-1]))        
    return return_dict

def cross_validation_sgd(model_initial, learning_rates, dataset, K, criterion, args):
    """
    Performs cross-validation for learning rate of SGD.

    Parameters
    ----------
    model_initial : torch.nn.Module
        model to train.
    learning_rates : list of float
        List of learning rates to try.
    dataset : dataset
        Dataset from which to do the cross-validation.
    K : int
        Number of folds.
    criterion :  torch.nn._Loss
        loss function.
    args : dict
        Dictionary with eperiment parameters.


    Returns
    -------
    return_dict : dict
        dict with results of cross-vallidation.

    """
    
    k_indices = build_k_indices(len(dataset), K, args['seed'])
    
    len_learning_rate = len(learning_rates)

    training_errors_during_training = np.zeros((len_learning_rate, K, args['epochs']))
    validation_errors_during_training = np.zeros((len_learning_rate, K, int(args['epochs']//args['log_interval']) ))
    
    training_accuracies_during_training = np.zeros((len_learning_rate, K, args['epochs']))
    validation_accuracies_during_training = np.zeros((len_learning_rate, K, int(args['epochs']//args['log_interval']) ))
    
    for i, lr in enumerate(learning_rates):
        print("Training for learning rate={}".format(lr))
        for j, k in enumerate(range(K)):
            print("Fold {}".format(k))
            model = copy.deepcopy(model_initial)
            trainloader, validationloader = create_train_val_dataloader(k, k_indices, dataset, args)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args['momentum'])

            train_losses, val_losses, train_accuracies, val_accuracies, _ =\
                train_loop_optimizer(model, trainloader, validationloader,
                                     optimizer, criterion, args['epochs'],
                                     args['log_interval'], args['cuda'])
            
            training_errors_during_training[i,j] = train_losses
            validation_errors_during_training[i,j] = val_losses
            training_accuracies_during_training[i,j] = train_accuracies
            validation_accuracies_during_training[i,j] = val_accuracies
    return_dict = build_return_dict_CV(training_errors_during_training,
                                       validation_errors_during_training,
                                       training_accuracies_during_training,
                                       validation_accuracies_during_training,
                                       "SGD lr:{}-{}".format(learning_rates[0], learning_rates[-1]))        
    return return_dict

def cross_validation_svrg(model_initial, learning_rates, dataset, K, criterion, args):
    """
    Performs cross-validation for learning rate of SVRG.

    Parameters
    ----------
    model_initial : torch.nn.Module
        model to train.
    learning_rates : list of float
        List of learning rates to try.
    dataset : dataset
        Dataset from which to do the cross-validation.
    K : int
        Number of folds.
    criterion :  torch.nn._Loss
        loss function.
    args : dict
        Dictionary with eperiment parameters.


    Returns
    -------
    return_dict : dict
        dict with results of cross-vallidation.

    """
    
    k_indices = build_k_indices(len(dataset), K, args['seed'])
    
    len_learning_rate = len(learning_rates)

    training_errors_during_training = np.zeros((len_learning_rate, K, args['epochs']))
    validation_errors_during_training = np.zeros((len_learning_rate, K, int(args['epochs']//args['log_interval']) ))
    
    training_accuracies_during_training = np.zeros((len_learning_rate, K, args['epochs']))
    validation_accuracies_during_training = np.zeros((len_learning_rate, K, int(args['epochs']//args['log_interval']) ))
    
    for i, lr in enumerate(learning_rates):
        print("Training for learning rate={}".format(lr))
        for j, k in enumerate(range(K)):
            print("Fold {}".format(k))
            model = copy.deepcopy(model_initial)
            trainloader, validationloader = create_train_val_dataloader(k, k_indices, dataset, args)
            
            batch_size = trainloader.batch_size    
            train_losses, val_losses, train_accuracies, val_accuracies, _, \
            _, _ = train_loop_SVRG(model, trainloader, validationloader, lr, 
                            freq = args['svrg_freq']*len(trainloader.dataset)/batch_size, 
                            criterion = criterion, epochs_to_run=args['epochs'],
                            log_interval=args['log_interval'], cuda=args['cuda'])
            
            training_errors_during_training[i,j] = train_losses
            validation_errors_during_training[i,j] = val_losses
            training_accuracies_during_training[i,j] = train_accuracies
            validation_accuracies_during_training[i,j] = val_accuracies
    return_dict = build_return_dict_CV(training_errors_during_training,
                                       validation_errors_during_training,
                                       training_accuracies_during_training,
                                       validation_accuracies_during_training,
                                       "SVRG lr:{}-{}".format(learning_rates[0], learning_rates[-1]))        
    return return_dict

def cross_validation_storm_c(model_initial, c_values, dataset, K, criterion, args):
    """
    Performs cross-validation for hyperparameter c of SVRG.

    Parameters
    ----------
    model_initial : torch.nn.Module
        model to train.
    c_values : list of float
        list of c values to try.
    dataset : dataset
        Dataset from which to do the cross-validation.
    K : int
        Number of folds.
    criterion :  torch.nn._Loss
        loss function.
    args : dict
        Dictionary with eperiment parameters.


    Returns
    -------
    return_dict : dict
        dict with results of cross-vallidation.

    """
    
    k_indices = build_k_indices(len(dataset), K, args['seed'])
    
    len_c_values = len(c_values)

    training_errors_during_training = np.zeros((len_c_values, K, args['epochs']))
    validation_errors_during_training = np.zeros((len_c_values, K, int(args['epochs']//args['log_interval']) ))
    
    training_accuracies_during_training = np.zeros((len_c_values, K, args['epochs']))
    validation_accuracies_during_training = np.zeros((len_c_values, K, int(args['epochs']//args['log_interval']) ))
    
    for i, c in enumerate(c_values):
        print("Training for c_value={}".format(c))
        for j, k in enumerate(range(K)):
            print("Fold {}".format(k))
            model = copy.deepcopy(model_initial)
            trainloader, validationloader = create_train_val_dataloader(k, k_indices, dataset, args)
            
            
            train_losses, val_losses, train_accuracies, val_accuracies, _ = \
            train_loop_storm_optim(model, trainloader, validationloader, k=args['k'],
                                   w=args['w'], c=c, criterion = criterion,
                                   epochs_to_run=args['epochs'], 
                                   log_interval=args['log_interval'], cuda=args['cuda'])
            
            training_errors_during_training[i,j] = train_losses
            validation_errors_during_training[i,j] = val_losses
            training_accuracies_during_training[i,j] = train_accuracies
            validation_accuracies_during_training[i,j] = val_accuracies
    return_dict = build_return_dict_CV(training_errors_during_training,
                                       validation_errors_during_training,
                                       training_accuracies_during_training,
                                       validation_accuracies_during_training,
                                       "STORM c:{}-{}".format(c_values[0], c_values[-1]))        
    return return_dict

def cross_validation_storm_k(model_initial, k_values, dataset, K, criterion, args):
    """
    Performs cross-validation for hyperparameter k of SVRG.

    Parameters
    ----------
    model_initial : torch.nn.Module
        model to train.
    k_values : list of float
        list of k values to try.
    dataset : dataset
        Dataset from which to do the cross-validation.
    K : int
        Number of folds.
    criterion :  torch.nn._Loss
        loss function.
    args : dict
        Dictionary with eperiment parameters.


    Returns
    -------
    return_dict : dict
        dict with results of cross-vallidation.

    """
    
    k_indices = build_k_indices(len(dataset), K, args['seed'])
    
    len_k_values = len(k_values)

    training_errors_during_training = np.zeros((len_k_values, K, args['epochs']))
    validation_errors_during_training = np.zeros((len_k_values, K, int(args['epochs']//args['log_interval']) ))
    
    training_accuracies_during_training = np.zeros((len_k_values, K, args['epochs']))
    validation_accuracies_during_training = np.zeros((len_k_values, K, int(args['epochs']//args['log_interval']) ))
    
    for i, k in enumerate(k_values):
        print("Training for k_value={}".format(k))
        for j, k in enumerate(range(K)):
            print("Fold {}".format(k))
            model = copy.deepcopy(model_initial)
            trainloader, validationloader = create_train_val_dataloader(k, k_indices, dataset, args)
            
            
            train_losses, val_losses, train_accuracies, val_accuracies, _ = \
            train_loop_storm_optim(model, trainloader, validationloader, k=k,
                                   w=args['w'], c=args['c'], criterion = criterion,
                                   epochs_to_run=args['epochs'], 
                                   log_interval=args['log_interval'], cuda=args['cuda'])
            
            training_errors_during_training[i,j] = train_losses
            validation_errors_during_training[i,j] = val_losses
            training_accuracies_during_training[i,j] = train_accuracies
            validation_accuracies_during_training[i,j] = val_accuracies
    return_dict = build_return_dict_CV(training_errors_during_training,
                                       validation_errors_during_training,
                                       training_accuracies_during_training,
                                       validation_accuracies_during_training,
                                       "STORM k:{}-{}".format(k_values[0], k_values[-1]))        
    return return_dict

def build_return_dict_CV(train_losses, validation_losses, train_accuracies,
                            validation_accuracies, description):
    """
    Creates a dictionary with the output of the train_loop with optimizer

    Parameters
    ----------
    train_losses : numpy array (K, learning_rate, epochs)
        train losses after each epoch.
    validation_losses : numpy array (K, learning_rate, epochs//log_interval)
        validation losses after each epoch.
    train_accuracies : numpy array (K, learning_rate, epochs)
        train accuracies after each epoch.
    validation_accuracies : numpy array (K, learning_rate, epochs//log_interval)
        validation accuracies after each epoch.
    description : str
        summary of the cross validation.

    Returns
    -------
    return_values : dict
        dictionary with mapped values.

    """
    return_values = {}
    return_values['train_losses']=train_losses
    return_values['validation_losses']=validation_losses
    return_values['train_accuracies']=train_accuracies
    return_values['validation_accuracies']=validation_accuracies
    return_values['description']=description
    return return_values       
        
        
    
    
    
    