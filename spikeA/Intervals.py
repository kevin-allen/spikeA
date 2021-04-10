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
    total_interval_duration_samples(): calculate the interval duration
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
            print("inter should be a numpy.ndarray but was {}".format(type(inter)))
        
        self.inter = inter
        self.sampling_rate = sampling_rate
       
        print("{} intervals, sampling rate: {}".format(self.inter.shape[0],self.sampling_rate))
    def total_interval_duration_samples(self):
        """
        Calculate the duration of the time in the intervals

        """
        return np.sum(self.inter[:,1]-self.inter[:,0])
    def total_interval_duration_seconds(self):
        return self.total_interval_duration_samples()/self.sampling_rate
