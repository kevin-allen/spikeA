"""
File containing the definition of the Spike_train class
"""

import numpy as np
import pandas as pd
from scipy import stats

class Spike_train:
    """
    Class representing the spikes recorded from several neurons.
    
    Attributes:
    name: name for the spike train
    st: list of 1d numpy arrays containing the spike time for each cluster
    cluster_ids: list of spike cluster IDs (integer). Each cluster or cell has a unique number.

    ifr: numpy array containing the instantaneous firing rate of clusters
    mean_rate: numpy array containing the mean firing rate of the clusters

    Methods:
    mean_firing_rate(): calculate the mean firing rate of the clusters
    instantaneous_firing_rate(): calculate the instantaneous firing rate of the cluster over time
    load_spike_train_from_files(): load the spike trains from files (e.g., res and clu file)
    """
    def __init__(self,name=None):
        """
        Constructor of the Spike_train class

        Arguments:
        name: Name for the Spike_train object
        """
        print(name)

    def mean_firing_rate(self):
        """
        Calculate the mean firing rate (number of spikes / sec) of each cluster

        The results are stored in a numpy array with the same dimension as cluster_ids array
        """
        
        pass
    def instantaneous_firing_rate(self):
        """
        Calculate the instantaneous firing rate. This is the firing rate of the neuron over time.

        The spikes are counted in equal sized time windows. 
        Then the spike count array is smooth with a gaussian kernel.
        The result is a 2D numpy array with the dimensions cluster and time_window.
        The results are saved in the attribute ifr
        """
        
        pass
    def load_spike_train_from_files(self,type="resClu"):
        """
        Load the spike trains from file

        Read the number of clusters from the clu file.
        Read the res file with the spike time.
        Create a list of numpy arrays, one array per cluster. Each array contains the spike time for one neuron.
        """
        pass
    
