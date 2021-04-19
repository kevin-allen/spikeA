"""
File containing the definition of the Dat_file_reader class
"""
import numpy as np
import pandas as pd
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
    def __init__(self,file_names,n_channels):
        """
        Constructor of the Dat_file_reader class

        Arguments:
        file_names: List containing the full path of the .dat files
        n_channels: number of channels in the files
        """

        # check that the n_channels make sense

        # make sure the files exist

        # get the file size

        # make sure the file size is a multiple of n_channels*2

        # get the number of samples per file
        
        
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
