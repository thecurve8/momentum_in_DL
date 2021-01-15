# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:31:24 2020

@author: Alexander

Contains methods to save and load results from experiments.

"""

import os
from os import listdir
from os.path import isfile, join
import pickle

def find_next_available_file_number(dir_algo, file_name):
    """
    Finds next available file number given an already specified file prefix

    Parameters
    ----------
    dir_algo : str
        Directory where the file will be saved.
    file_name : str
        Prefix of the name (number still missing) of the file that will be saved.

    Returns
    -------
    int
        Next available number for the given prefix.

    """
    onlypklfiles = [os.path.splitext(f)[0] for f in listdir(dir_algo) if (isfile(join(dir_algo, f)) and f.lower().endswith('.pkl'))]
    if len(onlypklfiles)==0:
        return 0
    filteredfiles = [f for f in onlypklfiles if f.startswith(file_name)]
    if len(filteredfiles)==0:
        return 0
    biggest_seen = -1
    for f in filteredfiles:
        x = f.split("_")
        if int(x[-1])>biggest_seen:
            biggest_seen = int(x[-1])
    return biggest_seen+1

def float_to_str(float_value):
    """
    Returns string mapping of a float.

    Parameters
    ----------
    float_value : float
        Float value to map.

    Returns
    -------
    str
        String mapping where the dot in the float is replaced by a ^.

    """
    return str(float_value).replace('.', '^')

def str_to_float(str_value):
    """
    Returns initial float value after string mapping.

    Parameters
    ----------
    str_value : str
        string mapping.

    Returns
    -------
    float
        Initial float value before the mapping.

    """
    return float(str_value.replace('^', '.'))

def create_name_beginning(algo, model_name, criterion_name, args):
    """
    Create a filenemae given a specified algorithm, model, criterion and arguments used for training.

    Parameters
    ----------
    algo : str
        Algorithm of the optimizer can be (SVRG, SGD, ADAM, STORM or ADAGRAD).
    model_name : str
        Name of the trained model.
    criterion_name : str
        Name of the criterion used during training.
    args : dict
        argument dictionary used during training.

    Raises
    ------
    NotImplementedError
        If algo is not in (SVRG, SGD, ADAM, STORM or ADAGRAD).

    Returns
    -------
    name : str
        Specified filename.
        SVRG : svrg_{model_name}_{epochs}_{lr}_{seed}_{criterion_name}_{svrg_freq}_
        SGD : sgd_{model_name}_{epochs}_{lr}_{seed}_{criterion_name}_{momentum}_
        ADAM : adam_{model_name}_{epochs}_{lr}_{seed}_{criterion_name}_
        STORM : svrg_{model_name}_{epochs}_{seed}_{criterion_name}_{k}_{w}_{c}_
        ADAGRAD : adagrad_{model_name}_{epochs}_{lr}_{seed}_{criterion_name}_
        
        All floats above are encoded with the float_to_str function.

    """
    
    if algo == 'SVRG':
        name = "svrg_" + model_name+"_"+str(args['epochs'])+ \
                "_"+float_to_str(args['lr'])+\
                "_"+str(args['seed'])+"_"+ criterion_name+"_" +\
                float_to_str(args['svrg_freq'])+"_"
        return name
    if algo == 'SGD':
        name = "sgd_" + model_name+"_"+str(args['epochs'])+ \
                "_"+float_to_str(args['lr'])+\
                "_"+str(args['seed'])+"_"+ criterion_name+"_" +\
                float_to_str(args['momentum'])+"_"
        return name
    if algo == 'ADAM':
        name = "adam_" + model_name+"_"+str(args['epochs'])+ \
                "_"+float_to_str(args['lr'])+\
                "_"+str(args['seed'])+"_"+ criterion_name+"_" 
        return name
    if algo == 'STORM':
        name = "storm_" + model_name+"_"+str(args['epochs'])+ \
                "_"+str(args['seed'])+"_"+ criterion_name+\
                "_"+str(args['k'])+"_"+ str(args['w'])+\
                "_"+str(args['c'])+"_"
        return name
    if algo == 'ADAGRAD':
        name = "adagrad_" + model_name+"_"+str(args['epochs'])+ \
                "_"+float_to_str(args['lr'])+\
                "_"+str(args['seed'])+"_"+ criterion_name+"_" 
        return name
    else:
        raise NotImplementedError("Nothing defined for algo name {}".format(algo))
        
def create_name_beginning_cv(algo, model_name, criterion_name, args, from_value, to_value):
    """
    Create file name for Cross validation

    Parameters
    ----------
    algo : str
        Algorithm of the optimizer can be (SVRG, SGD, ADAM, STORM or ADAGRAD).
    model_name : str
        Name of the trained model.
    criterion_name : str
        Name of the criterion used during training.
    args : dict
        argument dictionary used during training.
    from_value : float
        Smallest value of cross-validation.
    to_value : float
        Biggest value of cross-validation.

    Raises
    ------
    NotImplementedError
        If algo is not in (SVRG, SGD, ADAM, STORM or ADAGRAD).

    Returns
    -------
    name : str
        Specified filename.
        SVRG : svrg_{model_name}_{epochs}_{lr}_{seed}_{criterion_name}_{svrg_freq}_from_{from_value}_to_{to_value}_
        SGD : sgd_{model_name}_{epochs}_{lr}_{seed}_{criterion_name}_{momentum}_from_{from_value}_to_{to_value}_
        ADAM : adam_{model_name}_{epochs}_{lr}_{seed}_{criterion_name}_from_{from_value}_to_{to_value}_
        STORM : svrg_{model_name}_{epochs}_{seed}_{criterion_name}_{k}_{w}_{c}_from_{from_value}_to_{to_value}_
        ADAGRAD : adagrad_{model_name}_{epochs}_{lr}_{seed}_{criterion_name}_from_{from_value}_to_{to_value}_
        
        All floats above are encoded with the float_to_str function.

    """
    if algo == 'SVRG':
        name = "svrg_" + model_name+"_"+str(args['epochs'])+ \
                "_"+str(args['seed'])+"_"+ criterion_name+"_" +\
                float_to_str(args['svrg_freq'])+\
                "_"+"from"+str(from_value)+"to"+str(to_value)+"_"
        return name
    if algo == 'SGD':
        name = "sgd_" + model_name+"_"+str(args['epochs'])+ \
                "_"+str(args['seed'])+"_"+ criterion_name+"_" +\
                float_to_str(args['momentum'])+\
                "_"+"from"+str(from_value)+"to"+str(to_value)+"_"
        return name
    if algo == 'ADAM':
        name = "adam_" + model_name+"_"+str(args['epochs'])+ \
                "_"+str(args['seed'])+"_"+ criterion_name+\
                "_"+"from"+str(from_value)+"to"+str(to_value)+"_" 
        return name
    if algo == 'STORM':
        name = "storm_" + model_name+"_"+str(args['epochs'])+ \
                "_"+str(args['seed'])+"_"+ criterion_name+\
                "_"+str(args['k'])+"_"+ str(args['w'])+\
                "_"+str(args['c'])+\
                "_"+"from"+str(from_value)+"to"+str(to_value)+"_"
        return name
    if algo == 'ADAGRAD':
        name = "adagrad_" + model_name+"_"+str(args['epochs'])+ \
                "_"+str(args['seed'])+"_"+ criterion_name+\
                "_"+"from"+str(from_value)+"to"+str(to_value)+"_" 
        return name
    else:
        raise NotImplementedError("Nothing defined for algo name {}".format(algo))
    
def create_name(algo, model_name, criterion_name, args, dir_algo):
    """
    Creates path and full name with filenumber of a given experiment.

    Parameters
    ----------
    algo : str
        Algorithm of the optimizer can be (SVRG, SGD, ADAM, STORM or ADAGRAD).
    model_name : str
        Name of the trained model.
    criterion_name : str
        Name of the criterion used during training.
    args : dict
        argument dictionary used during training.
    dir_algo : str
        Directory where the file will be saved.

    Returns
    -------
    name : str
        path to file that will be created.

    """
    beginning_name = create_name_beginning(algo, model_name, criterion_name, args)
    file_number = find_next_available_file_number(dir_algo, beginning_name)
    name =beginning_name + str(file_number)+".pkl"
    return name

def create_name_cv(algo, model_name, criterion_name, args, dir_algo, from_value, to_value):
    """
    Creates the path and name of the file with filenumber from a cross-validation experiment.

    Parameters
    ----------
    algo : str
        Algorithm of the optimizer can be (SVRG, SGD, ADAM, STORM or ADAGRAD).
    model_name : str
        Name of the trained model.
    criterion_name : str
        Name of the criterion used during training.
    args : dict
        argument dictionary used during training.
    dir_algo : str
        Directory where the file will be saved.
    from_value : float
        Smallest value of cross-validation.
    to_value : float
        Biggest value of cross-validation.

    Returns
    -------
    name : str
        path to file that will be created..

    """
    
    beginning_name = create_name_beginning_cv(algo, model_name, criterion_name, args, from_value, to_value)
    file_number = find_next_available_file_number(dir_algo, beginning_name)
    name =beginning_name + str(file_number)+".pkl"
    return name

def check_name(algo, model_name, criterion_name, args, dir_algo, specified_file_number=-1):
    """
    Check whether a given experiment has been saved exists

    Parameters
    ----------
    algo : str
        Algorithm of the optimizer can be (SVRG, SGD, ADAM, STORM or ADAGRAD).
    model_name : str
        Name of the trained model.
    criterion_name : str
        Name of the criterion used during training.
    args : dict
        argument dictionary used during training.
    dir_algo : str
        Directory to check.
    specified_file_number : int, optional
        File number, if -1 take the biggest one that is found. The default is -1.

    Returns
    -------
    bool
        True if file exists.
    str
        full name of the file.

    """
    
    beginning_name = create_name_beginning(algo, model_name, criterion_name, args)
    #retrieve last file
    if specified_file_number == -1:
        file_number = find_next_available_file_number(dir_algo, beginning_name)
        if file_number == 0:
            return False, beginning_name+"xx"
        else : 
            return True, beginning_name + str(file_number-1)+".pkl"
    else:
        name = beginning_name + str(specified_file_number)+".pkl"
        full_path = os.path.join(dir_algo, name)
        return os.path.isfile(full_path), name
    
def save_metrics(return_dict, algo, model_name, criterion_name, args, dir_name = '/content/drive/My Drive/Semester_Project_MLO/saved/'):
    """
    Save given experiment into a pkl file.

    Parameters
    ----------
    return_dict : dict
        return_dict after the experiment.
    algo : str
        Algorithm of the optimizer can be (SVRG, SGD, ADAM, STORM or ADAGRAD)..
    model_name : str
        Name of the trained model.
    criterion_name : str
        Name of the criterion used during training..
    args : dict
        Arguments used during training.
    dir_name : str, optional
        Directory where to save the file. The default is '/content/drive/My Drive/Semester_Project_MLO/saved/'.

    Raises
    ------
    TypeError
        If given algo is not a string.
    ValueError
        If algo is not in (SVRG, SGD, ADAM, STORM or ADAGRAD).

    Returns
    -------
    None.

    """
    available_algo_names = ('SVRG', 'ADAM', 'SGD', 'STORM')
    if not isinstance(algo, str):
        raise TypeError("Expected str for algo. Got {}".format(type(algo)))
    if algo not in available_algo_names:
        raise ValueError("Expected algo value in "+ str(available_algo_names) +
                         " got {}".format(algo))

    dir_algo = os.path.join(dir_name, algo)
    
    file_name = create_name(algo, model_name, criterion_name, args, dir_algo)
    full_path = os.path.join(dir_algo, file_name)
    
    with open(full_path, 'wb') as file:
        pickle.dump(return_dict, file)

def save_cv(return_dict, algo, model_name, criterion_name, args, from_value, to_value, dir_name = '/content/drive/My Drive/Semester_Project_MLO/saved/'):
    """
    Save CV experiment metrics in pkl file.

    Parameters
    ----------
    return_dict : dict
        return_dict after the experiment.
    algo : str
        Algorithm of the optimizer can be (SVRG, SGD, ADAM, STORM or ADAGRAD)..
    model_name : str
        Name of the trained model.
    criterion_name : str
        Name of the criterion used during training..
    args : dict
        Arguments used during training.
    from_value : float
        Smallest value used in cross-validation.
    to_value : float
        Largest value used in cross-validation.
    dir_name : str, optional
        Directory where to save the file. The default is '/content/drive/My Drive/Semester_Project_MLO/saved/'.

    Raises
    ------
    TypeError
        If given algo is not a string.
    ValueError
        If algo is not in (SVRG, SGD, ADAM, STORM or ADAGRAD).

    Returns
    -------
    None.

    """
    
    available_algo_names = ('SVRG', 'ADAM', 'SGD', 'STORM', 'ADAGRAD')
    if not isinstance(algo, str):
        raise TypeError("Expected str for algo. Got {}".format(type(algo)))
    if algo not in available_algo_names:
        raise ValueError("Expected algo value in "+ str(available_algo_names) +
                         " got {}".format(algo))

    
    dir_algo = os.path.join(dir_name, algo)
    
    file_name = create_name_cv(algo, model_name, criterion_name, args, dir_algo, from_value, to_value)
    full_path = os.path.join(dir_algo, file_name)
    
    with open(full_path, 'wb') as file:
        pickle.dump(return_dict, file)

def load_metrics_dict(algo, model_name, criterion_name, args, specified_file_number=-1, dir_name = '/content/drive/My Drive/Semester_Project_MLO/saved/'):
    """
    Retrieves a saved metric file.

    Parameters
    ----------
    aalgo : str
        Algorithm of the optimizer can be (SVRG, SGD, ADAM, STORM or ADAGRAD)..
    model_name : str
        Name of the trained model.
    criterion_name : str
        Name of the criterion used during training..
    args : dict
        Arguments used during training.
    specified_file_number : int, optional
        File number, if -1 take the biggest one that is found. The default is -1.
    dir_name : str, optional
        Directory where to load the file. The default is '/content/drive/My Drive/Semester_Project_MLO/saved/'.

    Raises
    ------
    TypeError
        If given algo is not a string.
    ValueError
        If algo is not in (SVRG, SGD, ADAM, STORM or ADAGRAD).

    Returns
    -------
    dict
        returns metric dictionary if file with specified experiment exists.

    """
    
    available_algo_names = ('SVRG', 'ADAM', 'SGD', 'STORM', 'ADAGRAD')
    if not isinstance(algo, str):
        raise TypeError("Expected str for algo. Got {}".format(type(algo)))
    if algo not in available_algo_names:
        raise ValueError("Expected algo value in "+ str(available_algo_names) +
                         " got {}".format(algo))
    
    
    dir_algo = os.path.join(dir_name, algo)
    
    exists, file_name = check_name(algo, model_name, criterion_name, args, dir_algo,
                              specified_file_number=specified_file_number)
    if exists:    
        full_path = os.path.join(dir_algo, file_name)
        
        with open(full_path, 'rb') as f:
            return pickle.load(f)
    else:
        print("File {} does not exist in directory {}".format(file_name, dir_algo))
    