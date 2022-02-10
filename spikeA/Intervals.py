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
    
    The time is in seconds stored as np.float32
    
    Attributes:
    
    inter: 2D numpy array containing the start and end of each interval
    
    Methods:
    total_interval_duration_samples(), calculate the interval duration
    spike_train_within_intervals(), return a spike train containing only the spikes within the intervals
    """
    def __init__(self,inter):
        """
        Constructor of the Interval class

        Arguments:
        inter: 2D numpy array containing the start and end of each interval, one row per interval. Time is in seconds as np.float32
        sampling_rate: number of samples per seconds
        """
        self.set_inter(inter)
        #print("{} intervals".format(self.inter.shape[0]))
        
    def set_inter(self,inter):
        
        # check that inter is a numpy array
        if not isinstance(inter, np.ndarray):
            raise TypeError("inter argument should be a numpy.ndarray but was {}".format(type(inter)))
        # if inter consists of only a single interval, add the missing 1 dimension
        if inter.ndim == 1:
            inter = np.array([inter])
        # assert that inter is a list of which each element is an interval
        if inter.ndim != 2:
            raise TypeError("inter argument should have 2 dimensions")
        # check if it has 2 column
        if inter.shape[1] != 2:
            raise ValueError("inter argument should be a numpy array with 2 columns")
            
        # check that second values is larger than first
        if np.any(inter[:,1]-inter[:,0] < 0):
            raise ValueError("inter argument: second column values should be larger than first column values")
        
        self.inter = inter
        
        #print("Time in intervals: {} sec".format(self.total_interval_duration_seconds()))
        
        
    def total_interval_duration_seconds(self):
        """
        Calculate the duration of the time in the intervals in samples
        """
        return np.sum(self.inter[:,1]-self.inter[:,0])
    
    def spike_train_within_intervals(self, st):
        """
        Return a 1D numpy array containing only the spike times that are within the intervals
        
        Argument
        st, 1D numpy array containing spike times
        
        Return
        1D numpy array containing spike times within the intervals
        """
        return st[self.is_within_intervals(st)]

    def is_within_intervals(self,time):
        """
        Return a 1D numpy array containing True or False indicating whether the values in time are within the intervals
        
        Artument
        time: 1D numpy array containing times
        
        Return
        1D numpy array of boolean indicating whether the time points were within the intervals
        """
        within = np.zeros((time.shape[0],self.inter.shape[0]))
        
        for i in range(self.inter.shape[0]):
            s=self.inter[i,0]
            e=self.inter[i,1]
            within[:,i]=np.logical_and(time>=s, time<=e)
        
        
        return np.sum(within,axis=1)>0
        