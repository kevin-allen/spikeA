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
    size_of_files
    samples_per_file
    files_first_sample
    files_last_sample
    
    Methods:
    get_first_last_samples_each_file()
    get_data_one_block()
    get_block_start_end_within_files()
    read_one_block()

    """
    def __init__(self,file_names,n_channels): ##We don't need sampling rate in read dat class
        """
        Constructor of the Dat_file_reader class

        Arguments:
        file_names: List containing the full path of the .dat files
        n_channels: number of channels in the files
        sampling_rate: sampling rate of the data
        """
        
        # check that the n_channels make sense        
        if not isinstance(n_channels, int):
            raise TypeError("Number of channels should be an integer")
        if n_channels < 1 :
            raise ValueError("Number of channels should be larger than 0")
        if not isinstance(file_names, list):
            raise TypeError("file_names should be a list")
                   
        # assign argument of function to the object attributes
        self.n_channels = n_channels
        self.file_names = file_names
            
        # make sure the files exist
        for f in self.file_names:
            if os.path.isfile(f) is False:
                raise ValueError("The file {} is missing.".format(f))
                
        # get the file size
        self.size_of_files = np.array([os.path.getsize(f) for f in self.file_names])
        
        # make sure the file size is a multiple of n_channels*2
        for i in range(len(self.size_of_files)):
            if self.size_of_files[i] % (self.n_channels*2) != 0:
                raise ValueError("Size of file {} can not be devided by {}".format(self.file_names[i],n_channels*2) + ". Number of bytes should be a multiple of n_channels*2")

        # get the number of samples in each file, a sample contains all channels at a given time point
        self.samples_per_file = (self.size_of_files / (2*self.n_channels)).astype(int)
        
        # get the first and last sample of each file
        self.files_first_sample, self.files_last_sample = self.get_first_last_samples_each_file()
    
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    def get_first_last_samples_each_file(self):
        """
        Calculate what the first and last sample of a file is

        Arguments:

        Returns: 
        tuple containing 2 1D numpy arrays (first and last sample in each file)
        """
        files_last_sample = np.cumsum(self.samples_per_file)-1
        files_first_sample = np.insert(files_last_sample+1,0,0)[0:-1]
        return files_first_sample, files_last_sample

    
    def get_data_one_block(self,start_sample,end_sample,channels):
        """
        Method that the end user should use to get data from dat files.
        
        Arguments:
        start_sample: first sample to get
        end_sample: last sample to get
        channels: 1D numpy array with the channels you want to get
        
        Return:
        2D numpy array (dtype=int16) containing the data requested        
        """
        if start_sample >= end_sample:
            raise ValueError("start_sample should be smaller than last_sample")
            
        if start_sample < 0:
            raise ValueError("start_sample should not be a negative value")
            
        if end_sample > self.files_last_sample[-1]:
            raise ValueError("end_sample should not be larger than the total number of samples")
        
        if type(channels) is not np.ndarray:
            raise TypeError("channels should be a numpy.ndarray")
        if channels.ndim != 1:
            raise ValueError("channels should be an np.array of 1 dimension")
        
        samples_to_read=end_sample-start_sample
        
        # start and end points of reading operations
        f1,i1,f2,i2 = self.get_block_start_end_within_files(start_sample,end_sample)
        
        # return the data block returned by our self.read_one_block() method
        return self.read_one_block(f1,i1,f2,i2,samples_to_read,channels)
        

    def get_block_start_end_within_files(self,start_index,end_index):
        """
        Function to get the start and end of a block in our collection of .dat files

        This function is needed because we sometime have several .dat files and we need to know in which file(s) to read from.

        Arguments: 
        start_index: first sample to read
        end_index: last index to read

        Return:
        A tuple with start_file_no, start_index_within_file, end_file_no, end_nidex_within_file
        """
        # get the starting point of reading operation in dat files (start_file_no,start_index_within_file)
        start_file_no = np.where((start_index >=self.files_first_sample) &  (start_index <self.files_last_sample))[0].item()
        start_index_within_file = start_index - self.files_first_sample[start_file_no]

        # get the end point of reading operation in dat files (end_file_no, end_index_within_file)
        end_file_no = np.where((end_index >=self.files_first_sample) &  (end_index < self.files_last_sample))[0].item()
        end_index_within_file = end_index - self.files_first_sample[end_file_no]

        # return a tuple with start and end of reading operation
        return start_file_no, start_index_within_file, end_file_no, end_index_within_file
    

    def read_one_block(self, f1,i1,f2,i2,samples_to_read,channels):
        """
        Function to read one block of consecutive data

        It can read data from several .dat files if needed.
        
        User should use the get_data method

        Arguments:
        f1: index of the file in which the block starts (starts at 0)
        i1: first sample to read in the first file
        f2: index of the file in which the block ends
        i2: last sample to read in the last file
        samples_to_read: total number of samples to read in the block
        channels: 1D np.array with the channels to read (starts at 0)
        Return:
        2D array containing the data to return
        """

        if f1 == f2:
            print("can read the block from a single file")
            print("Read file: ",self.file_names[f1], " from ", i1, "to" , i2)
            my_mapped_file = np.memmap(self.file_names[f1], dtype = "int16", mode = "r",
                                       shape = (self.n_channels, self.samples_per_file[f1]), order = "F")                    
            my_block = my_mapped_file[channels,i1:i2]
        else:
            print ("read the block from several files")
            # allocate the memory for the whole block
            my_block = np.empty((self.n_channels,samples_to_read),dtype="int16") # something similar
            copied = 0
            for i in range(f1,f2+1): # loop through the .dat files
                print("reading from file ",i)
                my_mapped_file = np.memmap(self.file_names[i], dtype = "int16", mode = "r",
                                       shape = (self.n_channels, self.samples_per_file[i]), order = "F")                       
                if i == f1: # first file    
                    print("copy from ",i1," to the end of the file (",self.samples_per_file[i],") in file ",f1)
                    print("into 0 to ", self.samples_per_file[i]-i1)
                    print(self.samples_per_file[i]-i1)
                    my_block[channels, 0:(self.samples_per_file[i]-i1)] = my_mapped_file[channels,i1:self.samples_per_file[i]]
                    copied = copied + self.samples_per_file[i]-i1

                elif i == f2: # last file
                    print("copy from the begining of the file ",f2, " to ", i2)
                    print("copied:",copied, " of ", samples_to_read)
                    my_block[channels, copied:] = my_mapped_file[channels,0:i2]

                else: # files in the middle
                    print("read the entire file ",i)
                    my_block[channels, copied:(copied+self.samples_per_file[i])] = my_mapped_file[channels,:]
                    copied = copied + self.samples_per_file[i] 
        return my_block
    
    