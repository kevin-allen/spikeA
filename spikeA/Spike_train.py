"""
File containing the definition of the Spike_train class
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from spikeA.Intervals import Intervals
from scipy.ndimage import gaussian_filter1d
from scipy import signal

class Spike_train:
    """
    Class representing the spikes recorded from one neuron.
    
    The class does the analysis of spike train. A spike train is defined as the spike in time emitted by a single neuron
    
    Attributes:
    
    name: name for the spike train
    st: 1d numpy arrays containing the spike time a neuron. The time is in sample number.
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
        if st.ndim != 1: # it should be one neuron so one dimention
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
        
        Arguments
        rate_hz: Firing rate of the spike train
        sampling_Rate: sampling rate for the poisson process
        length_sec: length of the spike train (sampling process in the poisson distribution)
        
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
        st = np.nonzero(poisson.rvs(mu, size=length))[0] # np.nonzero returns a tuple of arrays, we get the first element of the tuple
       
        self.set_spike_train(sampling_rate=sampling_rate, st = st)
        
        
    def generate_modulated_poisson_spike_train(self,rate_hz=20, sampling_rate=20000, length_sec=2,modulation_hz = 10, modulation_depth = 0.5,bins_per_cycle=10):
        """
        Generate a spike train from a random poisson distribution in which the firing rate to follow a sine wave
        
        This can be used to test the method calculating a power spectrum from the instantaenous firing rate.
        
        Arguments
        rate_hz: Firing rate of the spike train
        sampling_Rate: sampling rate for the poisson process
        length_sec: length of the spike train (sampling process in the poisson distribution)
        modulation_hz: frequency of the modulation
        modulation_depth: depth of the firing rate modulation by the sine wave, 1 = will go from rate_hz*0 to rate_hz*2, 0.5 = will go from rate_hz*0.5 to rate_hz*1.5
        bins_per_cycle: how many times we are changing the firing rate frequency per cycle.
        
        Results are stored in self.st
        """   
        pass
        
    def n_spikes(self):
        """
        Return the number of spikes of the cluster
        """
        if self.st is None:
            raise ValueError("set the spike train before using Spike_train.n_spike()")
        return self.st.shape[0]
    
    def mean_firing_rate(self):
        """
        Calculate the mean firing rate (number of spikes / sec) of each cluster
        Use the total time in seconds from self.intervals

        Return the mean firing rate
        """
        if self.st is None:
            raise ValueError("set the spike train before using Spike_train.mean_firing_rate()")
        return self.n_spikes() / self.intervals.total_interval_duration_seconds()
        
    def inter_spike_intervals(self):
        """
        Calculate the inter spike intervals

        The results are stored in a 1D numpy called self.isi
        self.isi should have a length of len(self.st) -1
        """
        if self.st is None:
            raise ValueError("set the spike train before using Spike_train.inter_spike_intervals()")
        self.isi = np.diff(self.st)
        
    def inter_spike_intervals_histogram(self, bin_size_ms=5,max_time_ms=200, density= False):
        """
        Calculate an inter spike interval histogram
        Save in self.isi_histogram
        """
        self.inter_spike_intervals()
        isi_ms = self.isi/self.sampling_rate*1000
        self.isi_histogram = np.histogram(isi_ms, bins= np.arange(0,max_time_ms+bin_size_ms,bin_size_ms),density= density)

    def plot_inter_spike_interval_histogram(self):
        """
        Plot the inter spike interval histogram using matplotlib
        """
        pass
    
    def instantaneous_firing_rate(self,bin_size_ms = 1, sigma = 1):
        """
        Calculate the instantaneous firing rate. This is the firing rate of the neuron over time.

        The spikes are counted in equal sized time windows. (histogram)
        Then the spike count array is smooth with a gaussian kernel. (convolution)
        The result is a tuple containing the ifr, count and edges. These are 1D numpy arrays
        """    
        st_ms=self.st/self.sampling_rate*1000
        count, edges = np.histogram(st_ms, bins = np.arange(0, self.intervals.total_interval_duration_seconds() * 1000+bin_size_ms, bin_size_ms))
        ifr = gaussian_filter1d(count.astype(np.float32), sigma = sigma)
        self.ifr = ifr,count,edges
        
      
    def instantaneous_firing_rate_autocorrelation(self):
        """
        Calculate the autocorrelation of the instantaneous firing rate array (self.isi)
        
        Save the results in self.ifr_autocorrelation
        """
        pass
        
    def instantaneous_firing_rate_power_spectrum(self):
        """
        Calculate the power spectrum of the instantaneous firing rate array (self.isi)
        
        Save the results in self.ifr_power_spectrum
        """
        f, ps = signal.periodogram(self.isi)
        self.ifr_power_spectrum = f, ps
    