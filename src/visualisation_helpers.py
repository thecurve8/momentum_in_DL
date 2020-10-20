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