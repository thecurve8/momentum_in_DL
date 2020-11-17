# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:23:05 2020

@author: Alexander
"""
import matplotlib.pyplot as plt

def plot_metrics(dict_after_training, title, kind='both', metric='loss', period_name = 'Epoch',
                 first_index=0, last_value=-1, log_yscale=False):
    """
    Plots the train metric and optionally the test metric

    Parameters
    ----------
    train_losses : dict
        dictionary after training process
    title : str
        title of the plot
    kind : str, optional
        Which values to plot, 'both', 'test', 'train'. Default is 'both'
    metric : str, optional
        Name of the plotted metric 'loss' or 'accu', default: 'loss'
    period_name : str, optional
        Name of the period between each measurment of the metric, default: 'Epoch'
    first_index : int, optional
        index of the first vaue to plot. Default is 0
    last_inex : int, optional
        index of the last value to plot. Default is -1
    log_yscale : bool
        Default is False
    
    
    """
    if metric == 'loss':
        key = "_losses"
        plt.ylabel("Loss")
        
    if metric == 'accu':
        key = "_accuracies"
        plt.ylabel("Accuracy")
        
    if kind == 'both' or kind =='train':    
        plt.plot(dict_after_training['test'+key], 'b-', label='train')
        
    if kind == 'both' or kind == 'test':
        plt.plot(dict_after_training['train'+key], 'r-', label='test')
    plt.xlabel(period_name)
    if log_yscale:
        plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.show()
    
def plot_together(title, labels, styles, metric_dicts, metric = 'loss', kind='both', start_idx=0, end_idx=None, logyscale=False):
    """
    Plots the metrics from given dictionaries

    Parameters
    ----------
    title : str
        Title of the plot.
    labels : list of str
        labels of each dictionary.
    styles : list of str
        styles to uses. If same length as length of state_dicts, uses same style
        in case both train and test are plotted.
        If both train and test are plotted styles can be 2*len(state_dicts) and
        first half will be used for train metrics styles and second half for test
        metrics styles.
    metric_dicts : list of dict 
        List of dictionaries of metrics returned after training.
    metric : str, optional
        Metric to plot. Can be 'loss' or 'accu'. The default is 'loss'.
    kind : str, optional
        Kind of metric to plot. Can be 'test', 'train' or 'both'. The default is 'both'.
    start_idx : int, optional
        Index (usually epoch) from which to start plotting. The default is 0.
    end_idx : int, optional
        Index (usually epoch) until which to plot. If None is passed, all data will be plotted.
        The default is None.
    logyscale : bool, optional
        Logarithic y scale. The default is False.

    Returns
    -------
    None.

    """
    plt.title(title)
    if not end_idx:
        end_idx = len(metric_dicts[0]['train_losses'])
    x_axis = range(start_idx, end_idx)
    for i, dict_ in enumerate(metric_dicts):
        if metric=='loss':
            plt.ylabel("Loss")
            if kind == 'train' or kind == 'both':
                key = 'train_losses'
                plt.plot(x_axis, dict_[key][start_idx:end_idx], styles[i], label=labels[i]+' train')
            if kind == 'test' or kind == 'both':
                key = 'test_losses'
                style = styles[i]
                if kind == 'both' and len(styles) == 2*len(metric_dicts):
                    style = styles[len(metric_dicts) + i]
                plt.plot(x_axis, dict_[key][start_idx:end_idx], style, label=labels[i]+' test')
        
        if metric=='accu':
            plt.ylabel("Accuracy")
            if kind == 'train' or kind == 'both':
                key = 'train_accuracies'
                plt.plot(x_axis, dict_[key][start_idx:end_idx], styles[i], label=labels[i]+' train')
            if kind == 'test' or kind == 'both':
                key = 'test_accuracies'
                style = styles[i]
                if kind == 'both' and len(styles) == 2*len(metric_dicts):
                    style = styles[len(metric_dicts) + i]
                plt.plot(x_axis, dict_[key][start_idx:end_idx], style, label=labels[i]+' test')
    if logyscale:
        plt.yscale('log')    
    plt.legend()
    plt.show()
    
def compare_all(adam_return_dict, sgd_return_dict,svrg_return_dict, storm_return_dict):
    """
    Compare all metrics between ADAM, SGD, SVRG and STORM after training

    Parameters
    ----------
    adam_return_dict : dict
        Dictionary returned after training.
    sgd_return_dict : dict
        Dictionary returned after training.
    svrg_return_dict : dict
        Dictionary returned after training.
    storm_return_dict : dict
        Dictionary returned after training.

    Returns
    -------
    None.

    """
    fig, axs = plt.subplots(2,2,sharex=True, figsize=(12,12))
    adam_tr_acc = adam_return_dict['train_accuracies']
    sgd_tr_acc = sgd_return_dict['train_accuracies']
    svrg_tr_acc = svrg_return_dict['train_accuracies']
    storm_tr_acc = storm_return_dict['train_accuracies']

    adam_test_acc = adam_return_dict['test_accuracies']
    sgd_test_acc = sgd_return_dict['test_accuracies']
    svrg_test_acc = svrg_return_dict['test_accuracies']
    storm_test_acc = storm_return_dict['test_accuracies']

    adam_tr_loss = adam_return_dict['train_losses']
    sgd_tr_loss = sgd_return_dict['train_losses']
    svrg_tr_loss = svrg_return_dict['train_losses']
    storm_tr_loss = storm_return_dict['train_losses']

    adam_test_loss = adam_return_dict['test_losses']
    sgd_test_loss = sgd_return_dict['test_losses']
    svrg_test_loss = svrg_return_dict['test_losses']
    storm_test_loss = storm_return_dict['test_losses']

    axs[0,0].plot(adam_test_loss, label="Adam")
    axs[0,0].plot(sgd_test_loss, label="SGD")
    axs[0,0].plot(svrg_test_loss, label="SVRG")
    axs[0,0].plot(storm_test_loss, label="STORM")

    axs[0,1].plot(adam_tr_loss, label="Adam")
    axs[0,1].plot(sgd_tr_loss, label="SGD")
    axs[0,1].plot(svrg_tr_loss, label="SVRG")
    axs[0,1].plot(storm_tr_loss, label="STORM")

    axs[1,0].plot(adam_test_acc, label="Adam")
    axs[1,0].plot(sgd_test_acc, label="SGD")
    axs[1,0].plot(svrg_test_acc, label="SVRG")
    axs[1,0].plot(storm_test_acc, label="STORM")

    axs[1,1].plot(adam_tr_acc, label="Adam")
    axs[1,1].plot(sgd_tr_acc, label="SGD")
    axs[1,1].plot(svrg_tr_acc, label="SVRG")
    axs[1,1].plot(storm_tr_acc, label="STORM")

    axs[0,0].legend()
    axs[0,0].set_title("Test Losses comparison")
    axs[0,0].set_ylabel("Test loss")
    axs[0,0].set_yscale('log')
    axs[0,1].legend()
    axs[0,1].set_title("Training Losses comparison")
    axs[0,1].set_ylabel("Training loss")
    axs[0,1].set_yscale('log')
    axs[1,0].legend()
    axs[1,0].set_title("Test Accuracy comparsion")
    axs[1,0].set_ylabel("Test accuracy")
    axs[1,0].set_xlabel("Epochs")
    axs[1,1].legend()
    axs[1,1].set_title("Training Losses comparison")
    axs[1,1].set_ylabel("Training accuracy")
    axs[1,1].set_xlabel("Epochs")
    plt.show()

    
def compare_return_dicts(list_return_dicts, list_x_axis, list_names):
    if len(list_return_dicts) != len(list_x_axis):
        raise ValueError("The number of return_dict and x_axis is not the same")
    if len(list_return_dicts) != len(list_names):
        raise ValueError("The number of return_dict and names is not the same")
        
    fig, axs = plt.subplots(2,2,sharex=True, figsize=(12,12))
    for i, return_dict in enumerate(list_return_dicts):
        tr_acc = return_dict['train_accuracies']
        test_acc = return_dict['test_accuracies']
        tr_loss = return_dict['train_losses']
        test_loss = return_dict['test_losses']
        axs[0,0].plot(list_x_axis[i], test_loss, label=list_names[i])
        axs[0,1].plot(list_x_axis[i], tr_loss, label=list_names[i])
        axs[1,0].plot(list_x_axis[i], test_acc, label=list_names[i])
        axs[1,1].plot(list_x_axis[i], tr_acc, label=list_names[i])
    
    axs[0,0].legend()
    axs[0,0].set_title("Test Losses comparison")
    axs[0,0].set_ylabel("Test loss")
    axs[0,0].set_yscale('log')
    axs[0,1].legend()
    axs[0,1].set_title("Training Losses comparison")
    axs[0,1].set_ylabel("Training loss")
    axs[0,1].set_yscale('log')
    axs[1,0].legend()
    axs[1,0].set_title("Test Accuracy comparsion")
    axs[1,0].set_ylabel("Test accuracy")
    axs[1,0].set_xlabel("Gradient updates")
    axs[1,1].legend()
    axs[1,1].set_title("Training Losses comparison")
    axs[1,1].set_ylabel("Training accuracy")
    axs[1,1].set_xlabel("Gradient updates")
    plt.show()
    
def compare_all_with_gradient_updates(adam_return_dict, sgd_return_dict,svrg_return_dict, storm_return_dict,
                                      adam_x, sgd_x, svrg_x, storm_x):
    """
    Compare all metrics between ADAM, SGD, SVRG and STORM after training

    Parameters
    ----------
    adam_return_dict : dict
        Dictionary returned after training.
    sgd_return_dict : dict
        Dictionary returned after training.
    svrg_return_dict : dict
        Dictionary returned after training.
    storm_return_dict : dict
        Dictionary returned after training.
    adam_x : list 
        x axis coordinates for adam points
    sgd_x : list 
        x axis coordinates for sgd points
    svrg_x : list 
        x axis coordinates for svrg points
    storm_x : list 
        x axis coordinates for storm points
        

    Returns
    -------
    None.

    """
    fig, axs = plt.subplots(2,2,sharex=True, figsize=(12,12))
    adam_tr_acc = adam_return_dict['train_accuracies']
    sgd_tr_acc = sgd_return_dict['train_accuracies']
    svrg_tr_acc = svrg_return_dict['train_accuracies']
    storm_tr_acc = storm_return_dict['train_accuracies']

    adam_test_acc = adam_return_dict['test_accuracies']
    sgd_test_acc = sgd_return_dict['test_accuracies']
    svrg_test_acc = svrg_return_dict['test_accuracies']
    storm_test_acc = storm_return_dict['test_accuracies']

    adam_tr_loss = adam_return_dict['train_losses']
    sgd_tr_loss = sgd_return_dict['train_losses']
    svrg_tr_loss = svrg_return_dict['train_losses']
    storm_tr_loss = storm_return_dict['train_losses']

    adam_test_loss = adam_return_dict['test_losses']
    sgd_test_loss = sgd_return_dict['test_losses']
    svrg_test_loss = svrg_return_dict['test_losses']
    storm_test_loss = storm_return_dict['test_losses']

    axs[0,0].plot(adam_x, adam_test_loss, label="Adam")
    axs[0,0].plot(sgd_x, sgd_test_loss, label="SGD")
    axs[0,0].plot(svrg_x, svrg_test_loss, label="SVRG")
    axs[0,0].plot(storm_x, storm_test_loss, label="STORM")

    axs[0,1].plot(adam_x, adam_tr_loss, label="Adam")
    axs[0,1].plot(sgd_x, sgd_tr_loss, label="SGD")
    axs[0,1].plot(svrg_x, svrg_tr_loss, label="SVRG")
    axs[0,1].plot(storm_x, storm_tr_loss, label="STORM")

    axs[1,0].plot(adam_x, adam_test_acc, label="Adam")
    axs[1,0].plot(sgd_x, sgd_test_acc, label="SGD")
    axs[1,0].plot(svrg_x, svrg_test_acc, label="SVRG")
    axs[1,0].plot(storm_x, storm_test_acc, label="STORM")

    axs[1,1].plot(adam_x, adam_tr_acc, label="Adam")
    axs[1,1].plot(sgd_x, sgd_tr_acc, label="SGD")
    axs[1,1].plot(svrg_x, svrg_tr_acc, label="SVRG")
    axs[1,1].plot(storm_x, storm_tr_acc, label="STORM")

    axs[0,0].legend()
    axs[0,0].set_title("Test Losses comparison")
    axs[0,0].set_ylabel("Test loss")
    axs[0,0].set_yscale('log')
    axs[0,1].legend()
    axs[0,1].set_title("Training Losses comparison")
    axs[0,1].set_ylabel("Training loss")
    axs[0,1].set_yscale('log')
    axs[1,0].legend()
    axs[1,0].set_title("Test Accuracy comparsion")
    axs[1,0].set_ylabel("Test accuracy")
    axs[1,0].set_xlabel("Gradient updates")
    axs[1,1].legend()
    axs[1,1].set_title("Training Losses comparison")
    axs[1,1].set_ylabel("Training accuracy")
    axs[1,1].set_xlabel("Gradient updates")
    plt.show()
    
def compare_all_with_gradient_updates_without_storm(adam_return_dict, sgd_return_dict,svrg_return_dict,
                                      adam_x, sgd_x, svrg_x):
    """
    Compare all metrics between ADAM, SGD, SVRG after training

    Parameters
    ----------
    adam_return_dict : dict
        Dictionary returned after training.
    sgd_return_dict : dict
        Dictionary returned after training.
    svrg_return_dict : dict
        Dictionary returned after training.
    adam_x : list 
        x axis coordinates for adam points
    sgd_x : list 
        x axis coordinates for sgd points
    svrg_x : list 
        x axis coordinates for svrg points
        

    Returns
    -------
    None.

    """
    fig, axs = plt.subplots(2,2,sharex=True, figsize=(12,12))
    adam_tr_acc = adam_return_dict['train_accuracies']
    sgd_tr_acc = sgd_return_dict['train_accuracies']
    svrg_tr_acc = svrg_return_dict['train_accuracies']

    adam_test_acc = adam_return_dict['test_accuracies']
    sgd_test_acc = sgd_return_dict['test_accuracies']
    svrg_test_acc = svrg_return_dict['test_accuracies']

    adam_tr_loss = adam_return_dict['train_losses']
    sgd_tr_loss = sgd_return_dict['train_losses']
    svrg_tr_loss = svrg_return_dict['train_losses']

    adam_test_loss = adam_return_dict['test_losses']
    sgd_test_loss = sgd_return_dict['test_losses']
    svrg_test_loss = svrg_return_dict['test_losses']

    axs[0,0].plot(adam_x, adam_test_loss, label="Adam")
    axs[0,0].plot(sgd_x, sgd_test_loss, label="SGD")
    axs[0,0].plot(svrg_x, svrg_test_loss, label="SVRG")

    axs[0,1].plot(adam_x, adam_tr_loss, label="Adam")
    axs[0,1].plot(sgd_x, sgd_tr_loss, label="SGD")
    axs[0,1].plot(svrg_x, svrg_tr_loss, label="SVRG")

    axs[1,0].plot(adam_x, adam_test_acc, label="Adam")
    axs[1,0].plot(sgd_x, sgd_test_acc, label="SGD")
    axs[1,0].plot(svrg_x, svrg_test_acc, label="SVRG")

    axs[1,1].plot(adam_x, adam_tr_acc, label="Adam")
    axs[1,1].plot(sgd_x, sgd_tr_acc, label="SGD")
    axs[1,1].plot(svrg_x, svrg_tr_acc, label="SVRG")

    axs[0,0].legend()
    axs[0,0].set_title("Test Losses comparison")
    axs[0,0].set_ylabel("Test loss")
    axs[0,0].set_yscale('log')
    axs[0,1].legend()
    axs[0,1].set_title("Training Losses comparison")
    axs[0,1].set_ylabel("Training loss")
    axs[0,1].set_yscale('log')
    axs[1,0].legend()
    axs[1,0].set_title("Test Accuracy comparsion")
    axs[1,0].set_ylabel("Test accuracy")
    axs[1,0].set_xlabel("Gradient updates")
    axs[1,1].legend()
    axs[1,1].set_title("Training Losses comparison")
    axs[1,1].set_ylabel("Training accuracy")
    axs[1,1].set_xlabel("Gradient updates")
    plt.show()