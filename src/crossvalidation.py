# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:52:54 2020

@author: Alexander
"""
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from train_optimizer import train_loop_optimizer
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

def cross_validation_sgd(model, learning_rates, dataset, K, criterion, args):
    
    k_indices = build_k_indices(len(dataset), K, args['seed'])
    
    len_learning_rate = len(learning_rates)

    training_errors_during_training = np.zeros((K, len_learning_rate, args['epochs']))
    validation_errors_during_training = np.zeros((K, len_learning_rate, int(args['epochs']//args['log_interval']) ))
    
    training_accuracies_during_training = np.zeros((len_learning_rate, K, args['epochs']))
    validation_accuracies_during_training = np.zeros((len_learning_rate, K, int(args['epochs']//args['log_interval']) ))
    
    for i, lr in enumerate(learning_rates):
        for j, k in enumerate(range(K)):
            trainloader, validationloader = create_train_val_dataloader(k, k_indices, dataset, args)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args['momentum'])

            train_losses, val_losses, train_accuracies, val_accuracies, _ =\
                train_loop_optimizer(model, trainloader, validationloader,
                                     optimizer, criterion, args['epochs'],
                                     args['log_interval'], args['cuda'])
                
            training_errors_during_training[i,j]=train_losses
            validation_errors_during_training[i,j]=val_losses
            training_accuracies_during_training[i,j] = train_accuracies
            validation_accuracies_during_training[i,j] = val_accuracies
    return_dict = build_return_dict_CV(training_errors_during_training,
                                       validation_errors_during_training,
                                       training_accuracies_during_training,
                                       validation_accuracies_during_training,
                                       "CV sgd lr:{}-{}".format(learning_rates[0], learning_rates[-1]))
    
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
        
        
    
    
    
    