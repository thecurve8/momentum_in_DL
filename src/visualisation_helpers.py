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