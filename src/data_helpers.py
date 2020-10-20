# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:31:24 2020

@author: Alexander
"""
import os
from os import listdir
from os.path import isfile, join
import pickle

def find_next_available_file_number(dir_algo, file_name):
    onlypklfiles = [os.path.splitext(f)[0] for f in listdir(dir_algo) if (isfile(join(dir_algo, f)) and f.lower().endswith('.pkl'))]
    if len(onlypklfiles)==0:
        return 0
    filteredfiles = [f for f in onlypklfiles if f.startswith(file_name)]
    if len(filteredfiles)==0:
        return 0
    biggest_seen = -1
    for f in filteredfiles:
        x = f.split("_")
        if x[-1]>biggest_seen:
            biggest_seen = x[-1]
    return biggest_seen+1

def float_to_str(float_value):
    return str(float_value).replace('.', '^')

def str_to_float(str_value):
    return float(str_value.replace('^', '.'))

def create_name(algo, model_name, criterion_name, args, dir_algo):
    if algo == 'SVRG':
        name = "svrg_" + model_name+"_"+str(args['epochs'])+ \
                "_"+float_to_str(args['lr'])+\
                "_"+str(args['seed'])+"_"+ criterion_name+"_" +\
                float_to_str(args['svrg_freq'])+"_"
        file_number = find_next_available_file_number(dir_algo, name)
        name += str(file_number)+".pkl"
        return name
    if algo == 'SGD':
        name = "sgd_" + model_name+"_"+str(args['epochs'])+ \
                "_"+float_to_str(args['lr'])+\
                "_"+str(args['seed'])+"_"+ criterion_name+"_" +\
                float_to_str(args['momentum'])+"_"
        file_number = find_next_available_file_number(dir_algo, name)
        name += str(file_number)+".pkl"
        return name
    if algo == 'ADAM':
        name = "adam_" + model_name+"_"+str(args['epochs'])+ \
                "_"+float_to_str(args['lr'])+\
                "_"+str(args['seed'])+"_"+ criterion_name+"_" 
                
        file_number = find_next_available_file_number(dir_algo, name)
        name += str(file_number)+".pkl"
        return name
    else:
        raise NotImplementedError("Nothing defined for algo name {}".format(algo))
    
def save_metrics(return_dict, algo, model_name, criterion_name, args):
    
    available_algo_names = ('SVRG', 'ADAM', 'SGD')
    if not isinstance(algo, str):
        raise TypeError("Expected str for algo. Got {}".format(type(algo)))
    if algo not in available_algo_names:
        raise ValueError("Expected algo value in "+ str(available_algo_names) +
                         " got {}".format(algo))

    dir_name = '/content/drive/My Drive/Semester_Project_MLO/saved/'
    dir_algo = os.path.join(dir_name, algo)
    
    file_name = create_name(algo, model_name, criterion_name, args, dir_algo)
    full_path = os.path.join(dir_algo, file_name)
    
    with open(full_path, 'wb') as file:
        pickle.dump(return_dict, file)
        