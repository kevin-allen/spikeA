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
    def __init__(self,file_names,n_channels, sampling_rate): ##We don't need sampling rate in read dat class
        """
        Constructor of the Dat_file_reader class

        Arguments:
        file_names: List containing the full path of the .dat files
        n_channels: number of channels in the files
        sampling_rate: sampling rate of the data
        """
        
        # assign argument of function to the object attributes

        self.nchannels = n_channels
        self.file_names = file_names
        self.sampling_rate = sampling_rate
        self.sample_index = ()

        # check that the n_channels make sense
        
        if not isinstance(self.nchannels, int):
            raise TypeError("Number of channels should be an integer")

        # make sure the files exist
        
        for f in self.file_names:
            
            exist = os.path.isfile(f)
            if exist is False:
                raise ValueError("The file {} is missing.".format(f))
        print("All files are here")
                
        # get the file size
        
       
        self.size_of_files = [os.path.getsize(f) for f in self.file_names]
        

        # make sure the file size is a multiple of n_channels*2
        tmp = np.array(self.size_of_files) % (self.nchannels*2)
        if sum(tmp != 0) > 1:
            raise ValueError("Size can not be devided by {}".format(n_channels) + ". Number of bytes doesn't match the number of channes")

        # get the number of samples in each file
        
        self.sample_number_per_file = np.array(self.size_of_files) / 2
        
        # refer the sample number to an index that reflex the continueous sample number

        
        for k in range(0,len(self.file_names)):
            if k == 0:
                start = 0
                end = self.size_of_files[k]
            elif k >= 1:
                start = 0 + self.size_of_files[k-1]
                end = start + self.size_of_files[k]
            
            tmp = (self.file_names[k], start, end)
            
            self.sample_index = self.sample_index + tmp
        self.sample_index = np.reshape(self.sample_index, (len(self.file_names),3))

       
    def __str__(self):
        
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    def read_data_blocks(self,channels,start_samples,end_samples):
        """
        Read data blocks from the dat files

        Arguments
        channels: np.array containing the channels to get
        start_samples: np.array containing the start sample for each block
        end_samples: np.array containing the corresponding end sample (to the start sample). The length of end_samples and start_samples should be the same

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
        # find out in which files we need to get the data from (# Make an index of the start and end of each file)
        
        # if in a single file, get the data in one go
        # if in many file, create a loop to get the data in several steps
        
        if np.any(channels) >= self.nchannels:  #np.any(channels >= self.nchannels): # And negative numbers
            raise ValueError("The channel number is not in {}".format(range(0,self.nchannels-1)))
            
        df = np.empty((self.nchannels, 1))
        for i in range(len(self.file_names)): 
            tmp = np.memmap(self.file_names[i], dtype = "int16", mode = "r", 
                                 shape = (self.nchannels, int(self.size_of_files[i]/(2*self.nchannels))), order = "F")
            df = np.concatenate((df,tmp), axis = 1)  #Create a df that is already the size we want to have and fill them in.
            #int(self.size_of_files[i]/(2*self.nchannels))
        dff = df[channels, start_sample:start_sample + n_samples]
        return dff
    
    def which_dat(self, start_samples, end_samples):
        """
        Determine which .dat file to read based on the start_samples and the end_samples
        
        Arguments
        start_samples: np.array containing the first sample number of each block
        end_samples: np.array containing the last sample number of each block. 
        The length of these two arguments should be the same
        
        Return
        The name of the .dat file that should be used to access those sample blocks
        """
        
        #### Will it be better to read the whole length of all the .dat file and pick the time interval that we want? The critical point is that whether it's possible to read only
        #### a segment of the .dat file. What I did with read_one_block is that I read everything and then pick out the specified segment.
    