# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:23:05 2020

@author: Alexander
"""
import matplotlib.pyplot as plt

def plot_metrics(train_metric, test_metric=None, metric_name='Loss', period_name = 'Epoch'):
    """Plots the train metric and optionally the test metric

    Parameters
    ----------
    train_losses : list
        List of losses during training
    test_losses : list, optional
        List of test losses, if None nothing will be displayed, default: None
    metric_name : str, optional
        Name of the plotted metric, default: 'Loss'
    period_name : str, optional
        Name of the period between each measurment of the metric, default: 'Epoch'
    """
    
    plt.plot(train_metric, 'b-', label='train')
    if test_metric:
        plt.plot(test_metric, 'r-', label='test')
    plt.xlabel(period_name)
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()