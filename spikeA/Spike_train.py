"""
File containing the definition of the Spike_train class
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
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

        # assign argument of function to the object attributes
        self.name = name
        
        # if a spike train was passed to the constructor, set the spike train
        if st is not None:
            self.set_spike_train(sampling_rate=sampling_rate,st=st)
        else:
            self.st = None # we can use this to know if st has been set or not (whether it is None or none)
            
    
    def set_spike_train(self,sampling_rate,st):
        """
        Set the st and sampling_rate attribute of the Spike_train object
        
        Arguments
        sampling_rate
        st: 1D numpy array containing spike times in sample values
        """
        # check that st is a numpy array of 1 dimension
        if not isinstance(st, np.ndarray):
            raise TypeError("st argument of the Spike_time constructor should be a numpy.ndarray but was {}".format(type(st)))
        if st.ndim != 1:
            raise ValueError("st arguemnt of the Spike_time constructor should be a numpy array with 1 dimension but had {}".format(st.ndim))
        # check that sampling_rate value makes sense
        if sampling_rate <= 0 or sampling_rate > 100000:
            raise ValueError("sampling_rate arguemnt of the Spike_time constructor should be larger than 0 and smaller than 100000 but was {}".format(sampling_rate))
        
        self.st = st
        self.sampling_rate = sampling_rate
        # set default time intervals from 0 to just after the last spike
        self.intervals = Intervals(inter=np.array([[0,self.st.max()+1]]),
                                           sampling_rate=self.sampling_rate)
        
        print("Spike_train, name: {}, sampling rate: {} Hz, number of spikes {}".format(self.name,
                                                                                      self.sampling_rate,
                                                                                      self.st.shape[0]))
        print("Total interval time: {} sec".format(self.intervals.total_interval_duration_seconds()))
        
        
    def generate_poisson_spike_train(self,rate_hz=20, sampling_rate=20000, length_sec=2):
        """
        Generate a spike train from a random poisson distribution
        
        Results are stored in self.st
        """
        # check that sampling_rate value makes sense
        if sampling_rate <= 0 or sampling_rate > 100000:
            raise ValueError("sampling_rate arguemnt of the Spike_time constructor should be larger than 0 and smaller than 100000 but was {}".format(sampling_rate))
        # check that sampling_rate value makes sense
        if rate_hz < 0:
            raise ValueError("rate_hz argument of Spike_train.generate_poisson_spike_train() should not be negative but was {}".format(rate_hz))
        if length_sec < 0:
            raise ValueError("length_sec argument of Spike_train.generate_poisson_spike_train() should not be negative but was {}".format(length_sec))
        
        # variables to sample the poisson distribution
        length = sampling_rate*length_sec
        mu = rate_hz/sampling_rate # rate for each sample from the poisson distribution
        st = np.nonzero(poisson.rvs(mu, size=length))[0] # np.nonzero returns a tuple of arrays
       
        #print("Generating poisson spike train")
        #print("length: {}, mu: {}".format(length,mu))
        self.set_spike_train(sampling_rate=sampling_rate, st = st)
        
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
        return self.n_spikes() / self.intervals.total_interval_duration_seconds()
    
    def inter_spike_intervals(self):
        """
        Calculate the inter spike intervals

        The results are stored in a 1D numpy called self.isi
        self.isi should have a length of len(self.st) -1
        """
        self.isi = np.diff(self.st)
        
    def inter_spike_intervals_histogram(bin_size_ms=1,max_time_ms=10):
        """
        Calculate an inter spike interval histogram
        Save in self.isi_histogram
        """
        
    
    def instantaneous_firing_rate(self,bin_size_ms):
        """
        Calculate the instantaneous firing rate. This is the firing rate of the neuron over time.

        The spikes are counted in equal sized time windows. (histogram)
        Then the spike count array is smooth with a gaussian kernel. (convolution)
        The result is a 1D numpy array called self.ifr with the firing rate per time window
        """    
        pass
    
    def instantaneous_firing_rate_autocorrelation(self):
        """
        Calculate the autocorrelation of the instantaneous firing rate array (self.isi)
        
        Save the results in self.ifr_autocorrelation
        """