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
    total_interval_duration_samples
    total_interval_duration_seconds

    Methods:
    total_interval_duration(): calculate the interval duration
    """
    def __init__(self,inter,sampling_rate=20000):
        """
        Constructor of the Interval class

        Arguments:
        inter: 2D numpy array containing the start and end of each interval, one row per interval
        sampling_rate: number of samples per seconds
        """
        self.inter = inter
        self.sampling_rate = sampling_rate
       
        print("Intervals, sampling rate: {}".format(self.sampling_rate))
        print(self.inter)

    def total_interval_duration(self):
        """
        Calculate the duration of the time in the intervals

        """
        pass
