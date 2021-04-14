"""
File containing the definition of the Spike_train class
"""
import numpy as np
import pandas as pd
from scipy import stats

class Intervals:
    """
    Class representing time intervals
    
    Intervals are used to limit analysis to some time intervals in the recording. Instead of analyzing all the data, only some time windows are considered.
    
    Attributes:
    
    inter: 2D numpy array containing the start and end of each interval
    sampling_rate: number of samples per second
    
    Methods:
    total_interval_duration_samples(), calculate the interval duration
    spike_train_within_intervals(), return a spike train containing only the spikes within the intervals
    """
    def __init__(self,inter,sampling_rate=20000):
        """
        Constructor of the Interval class

        Arguments:
        inter: 2D numpy array containing the start and end of each interval, one row per interval
        sampling_rate: number of samples per seconds
        """
        # check that inter is a numpy array
        if not isinstance(inter, np.ndarray):
            raise TypeError("inter argument of the Interval constructor should be a numpy.ndarray but was {}".format(type(inter)))
        
        self.inter = inter
        self.sampling_rate = sampling_rate
        self.inter_ms = self.inter/self.sampling_rate*1000.0 
        
        #print("{} intervals, sampling rate: {}".format(self.inter.shape[0],self.sampling_rate))
    def total_interval_duration_samples(self):
        """
        Calculate the duration of the time in the intervals in samples
        """
        return np.sum(self.inter[:,1]-self.inter[:,0])
    def total_interval_duration_seconds(self):
        """
        Return the total interval duration in seconds
        """
        return self.total_interval_duration_samples()/self.sampling_rate
    
    def spike_train_within_intervals(self, st):
        """
        Return a 1D numpy array containing only the spike times that are within the intervals
        
        Argument
        st, 1D numpy array containing spike times
        
        Return
        1D numpy array containing spike times within the intervals
        """
        
        # this for loop is probably not the fastest
        # Ideally, we learn to do this with numpy function
        to_keep = np.empty_like(st,dtype=np.bool)
        for i, s in enumerate(st):
            to_keep[i] = np.any((self.inter[:,0] <= s) & (s <=self.inter[:,1]))
        return st[to_keep]

    def instantaneous_firing_rate_within_intervals(self, ifr, bin_size_ms)
        """
        Return a 1D numpy array containing only the rate values that are within the intervals
        
        self.inter_ms contains intervals in ms that can be used to establish if a bin is within the intervals.
        
        Argument
        ifr, 1D numpy array containing the instantaneous firing rate
        bin_size_ms, ms per bin
        ...
        """
        pass