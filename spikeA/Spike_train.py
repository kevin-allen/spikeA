"""
File containing the definition of the Spike_train class
"""
import numpy as np
import pandas as pd
from scipy import stats
from spikeA.Intervals import Intervals

class Spike_train:
    """
    Class representing the spikes recorded from one neuron.
    
    The class does the analysis of spike train. A spike train is defined as the spike in time emitted by a single neuron
    
    Attributes:
    
    name: name for the spike train
    st: 1d numpy arrays containing the spike time a neuron
    ifr: 1d numpy array containing the instantaneous firing rate of the neuron
    isi: 1d numpy array containing the inter-spike intervals between subsequent spikes
    
    sta: 1d numpy array containing the spike-time autocorrelation of the neuron
    m_rate: mean firing rate of the neuron

    Methods:
    n_spikes(): return the number of spikes of the neuron
    mean_firing_rate(): return the mean firing rate of the neuron
    inter_spike_intervals(): calculate the inter-spike intervals of the neuron
    instantaneous_firing_rate(): calculate the instantaneous firing rate in time
    """
    def __init__(self,name=None, st = None, sampling_rate=20000):
        """
        Constructor of the Spike_train class

        Arguments:
        name: Name for the Spike_train object
        st: 1d numpy array with the spike time of one neuron. Values are in samples
        sampling_rate: sampling rate in samples per second (Hz) of the recording system
        """
        
        # check that st is a numpy array of 1 dimension
        if not isinstance(st, np.ndarray):
            raise TypeError("st argument of the Spike_time constructor should be a numpy.ndarray but was {}".format(type(st)))
        if st.ndim != 1:
            raise ValueError("st arguemnt of the Spike_time constructor should be a numpy array with 1 dimension but had {}".format(st.ndim))
        # check that sampling_rate value makes sense
        if sampling_rate <= 0 or sampling_rate > 100000:
            raise ValueError("sampling_rate arguemnt of the Spike_time constructor should be larger than 0 and smaller than 100000 but was {}".format(sampling_rate))
        

        # assign argument of function to the object attributes
        self.name = name
        self.st = st
        self.sampling_rate = sampling_rate
        
        # set default time intervals from 0 to just after the last spike
        self.intervals = Intervals(inter=np.array([[0,self.st.max()+1]]),
                                   sampling_rate=self.sampling_rate)
        
        print("Spike_train, name: {}, sampling rate: {} Hz, number of spikes {}".format(self.name,
                                                                                  self.sampling_rate,
                                                                                  self.st.shape[0]))
        print("Total interval time: {} sec".format(self.intervals.total_interval_duration_seconds()))
        
    def n_spikes(self):
        """
        Return the number of spikes of the cluster
        """
        return self.st.shape[0]
    
    def mean_firing_rate(self):
        """
        Calculate the mean firing rate (number of spikes / sec) of each cluster
        Use the total time in seconds from self.intervals

        Return the mean firing rate
        """
        pass
    
    def inter_spike_intervals(self):
        """
        Calculate the inter spike intervals

        The results are stored in a 1D numpy called self.isi
        self.isi should have a length of len(self.st) -1
        """
        pass
    
    def instantaneous_firing_rate(self):
        """
        Calculate the instantaneous firing rate. This is the firing rate of the neuron over time.

        The spikes are counted in equal sized time windows. (histogram)
        Then the spike count array is smooth with a gaussian kernel. (convolution)
        The result is a 1D numpy array called self.ifr with the firing rate per time window
        """    
        pass
    
