"""
File containing the definition of the Dat_file_reader class
"""
import numpy as np
import pandas as pd
import os
from scipy import stats

class Dat_file_reader:
    """
    Class representing dat files
    
    It provides methods to read from a .dat file or a group of .dat files.

        
    Attributes:
    
    file_names: List of the file name (full path)
    n_channels: Number of channels in the files
    
    Methods:
    read_data_blocks, retrun 3D np array with the data from the blocks
    read_one_block, return 2D np array with the data from one block

    """
    def __init__(self,session_name,n_channels, sampling_rate):
        """
        Constructor of the Dat_file_reader class

        Arguments:
        session_name: List containing the full path of the .dat files
        n_channels: number of channels in the files
        sampling_rate: sampling rate of the data
        """
        
        # assign argument of function to the object attributes

        self.nchannels = n_channels
        self.session = session_name
        self.sampling_rate = sampling_rate

        # check that the n_channels make sense
        
        if isinstance(self.nchannels, float):
            raise ValueError("Number of channels should be integer but had float numbers")

        # make sure the files exist
        
        for f in range(0,len(self.session)):
            
            exist = os.path.isfile(self.session[f])
            if exist is False:
                raise ValueError("File not exist")
        print("All files are here")
                
        # get the file size
        
        size_of_files = []
        for f in range(0,len(self.session)):
            tmp = str(self.session[f])
            size_of_files = size_of_files + [os.path.getsize(tmp)]
            
        self.size_of_files = np.array(size_of_files)

        

        # make sure the file size is a multiple of n_channels*2
        
        tmp = self.size_of_files % n_channels*2
        if sum(tmp != 0) > 1:
            raise ValueError("Size can not be devided by {}".format(n_channels) + ". Number of bytes doesn't match the number of channes")

        # get the number of samples per channel in each file
        
        self.sample_number_per_file = self.size_of_files / (2*self.nchannels)
       
        
    def read_data_blocks(self,channels,start_samples,n_samples):
        """
        Read data blocks from the dat files

        Arguments
        channels: np.array containing the channels to get
        start_samples: np.array containing the start sample for each block
        n_samples: how many samples per block

        Return
        3D np.array containing the blocks of data from the dat files
        """
        # create the 3D np.array
        # loop and retrieve the individual blocks
        pass
    
  

    
    def read_one_block(self,channels,start_sample,n_samples):
        """
        Read one block of continuous samples in the dat file

        Arguments
        channels: np.array containing the channels to read
        start_sample: sample index at which we start getting the data
        n_samples: number of samples to get from the files

        Return
        2D np.array containing a single block of data from the dat files
        """
        # find out in which files we need to get the data from
        # if in a single file, get the data in one go
        # if in many file, create a loop to get the data in several steps
        pass
