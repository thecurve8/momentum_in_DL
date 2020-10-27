# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:52:54 2020

@author: Alexander
"""
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from train_optimizer import train_loop_optimizer

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

def cross_validation_adam(model, learning_rates, dataset, K, criterion, args):
    
    k_indices = build_k_indices(len(dataset), K, args['seed'])
    
    len_learning_rate = len(learning_rates)

    training_errors_during_training = np.zeros((K, len_learning_rate, args['epochs']))
    validation_errors_during_training = np.zeros((K, len_learning_rate, int(args['epochs']//args['log_interval']) ))
    
    for i, lr in enumerate(learning_rates):
        print("Training for learning rate={}".format(lr))
        for j, k in enumerate(range(K)):
            print("Fold {}".format(k))
            trainloader, validationloader = create_train_val_dataloader(k, k_indices, dataset, args)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            train_losses, val_losses, train_accuracies, val_accuracies, _ =\
                train_loop_optimizer(model, trainloader, validationloader,
                                     optimizer, criterion, args['epochs'],
                                     args['log_interval'], args['cuda'])
            training_errors_during_training[i,j]=train_losses
            validation_errors_during_training[i,j]=val_losses
            
    return training_errors_during_training, validation_errors_during_training

def cross_validation_sgd(model, learning_rates, dataset, K, criterion, args):
    
    k_indices = build_k_indices(len(dataset), K, args['seed'])
    
    len_learning_rate = len(learning_rates)

    training_errors_during_training = np.zeros((K, len_learning_rate, args['epochs']))
    validation_errors_during_training = np.zeros((K, len_learning_rate, int(args['epochs']//args['log_interval']) ))
    
    for i, lr in enumerate(learning_rates):
        for j, k in enumerate(range(K)):
            trainloader, validationloader = create_train_val_dataloader(k, k_indices, dataset, args)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            train_losses, val_losses, train_accuracies, val_accuracies, _ =\
                train_loop_optimizer(model, trainloader, validationloader,
                                     optimizer, criterion, args['epochs'],
                                     args['log_interval'], args['cuda'])
            training_errors_during_training[i,j]=train_losses
            validation_errors_during_training[i,j]=val_losses
            
    return training_errors_during_training, validation_errors_during_training           
        
        
    
    
    
    