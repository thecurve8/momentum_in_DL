# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:31:24 2020

@author: Alexander
"""
import os
from os import listdir
from os.path import isfile, join
import pickle

def find_next_available_file_number(dir):
    onlypklfiles = [os.path.splitext(f)[0] for f in listdir(dir) if (isfile(join(dir, f)) and f.lower().endswith('.pkl'))]
    return len(onlypklfiles)
    
def save_metrics(train_losses, test_losses, train_accuracies, test_accuracies,
                 model, optimizer=None, snapshot_model=None,
                 curr_batch_iter=None, algo='SVRG'):
    
    available_algo_names = ('SVRG', 'ADAM', 'SGD')
    if not isinstance(algo, str):
        raise TypeError("Expected str for algo. Got {}".format(type(algo)))
    if algo not in available_algo_names:
        raise ValueError("Expected algo value in "+ str(available_algo_names) +
                         " got {}".format(algo))

    dir_name = '/content/drive/My Drive/Semester Project MLO/saved/'
    dir_algo = os.path.join(dir_name, algo)
    last_file = find_next_available_file_number(dir_algo)
    full_path = os.path.join(dir_algo, algo + '_stats' + str(last_file+1) +'.pkl')
    with open(full_path, 'wb') as file:
        if algo == 'SVRG':
            pickle.dump([train_losses, test_losses, train_accuracies, test_accuracies, model, snapshot_model, curr_batch_iter], file)
        else:
            pickle.dump([train_losses, test_losses, train_accuracies, test_accuracies, model, optimizer], file)