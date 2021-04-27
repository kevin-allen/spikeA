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

        self.sample_index_per_channel = ()
        for k in range(0,len(self.file_names)):
        
            start = int((0 + sum(self.size_of_files[:k]))/(2*self.nchannels))
            end = int(start + (self.size_of_files[k]/(2*self.nchannels)) -1)
            
            tmp = (self.file_names[k], start, end, end-start)
            
            self.sample_index_per_channel = self.sample_index_per_channel + tmp
            
        self.sample_index_per_channel = np.reshape(self.sample_index_per_channel, (len(self.file_names),4))

       
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
        
        which_file = self.which_dat()
        acc = [which_file[i][3] for i in range(0,len(which_file))]
        total_len = np.cumsum(acc)
        df = np.empty((len(channels), total_len[-1]))                        # which_file retruns [file_name, start_sample, end_sample, n_samples] #accumulation of n_samples
        for k in range(0,len(which_file)):
            start = which_file[k][1]
            n_samples = which_file[k][3]
            df_tmp = self.read_one_block(file_base = which_file[k][0], channels = channels, start_sample = which_file[k][1], end_sample = which_file[k][2])
            df[:, total_len[k]-which_file[k][2]:total_len[k]] = df_tmp
        return df
 
    
    def read_one_block(self,file_base, channels,start_sample,end_sample):
        """
        Read one block of continuous samples in the dat file

        Arguments
        file_base: the dat file to read
        channels: np.array containing the channels to read
        start_sample: sample index at which we start getting the data
        end_sample: last sample index

        Return
        2D np.array containing a single block of data from the dat file
        """
        # find out in which files we need to get the data from (# Make an index of the start and end of each file)
        
        # if in a single file, get the data in one go
        # if in many file, create a loop to get the data in several steps
        
        if all(t not in range(0, self.nchannels) for t in channels):  #np.any(channels >= self.nchannels): # And negative numbers
            raise ValueError("The channel number is not in {}".format(range(0,self.nchannels)))
            
        n_samples = end_sample - start_sample    
        df = np.empty((len(channels), n_samples))
        index = self.file_names.index(file_base)
        tmp = np.memmap(file_base, dtype = "int16", mode = "r", 
                        shape = (self.nchannels, int(self.size_of_files[index]/(2*self.nchannels))), order = "F")
        tmp = tmp[channels, start_sample:end_sample]
        
        return tmp
    
    def which_file(self):
        """
        Determine which .dat file to read based on the start_samples and the end_samples, if two files are involved in one block, then make two blocks for reading
        
        Arguments
        start_samples: np.array containing the first sample number of each block
        end_samples: np.array containing the last sample number of each block. 
        The length of these two arguments should be the same
        
        Return
        A tuple containing the name of the .dat file that should be used to access each sample blocks, the start and end sample number, and n_samples of each block.
        """
        return [('/Users/t.yen/Desktop/mn2740-15042021_06B.dat', 0, 100000, 100000), ('/Users/t.yen/Desktop/mn2740-15042021_06B.dat', 0, 200000, 200000)]#, ('/Users/t.yen/Desktop/mn2740-15042021_06B.dat', 0, 200000, 200000)]
        #### Will it be better to read the whole length of all the .dat file and pick the time interval that we want? The critical point is that whether it's possible to read only
        #### a segment of the .dat file. What I did with read_one_block is that I read everything and then pick out the specified segment.
        
    def which_dat(self):
        """
        Determine which .dat file to read based on the start_samples and the end_samples, if two files are involved in one block, then make two blocks for reading
        
        Arguments
        start_samples: np.array containing the first sample number of each block
        end_samples: np.array containing the last sample number of each block. 
        The length of these two arguments should be the same
        
        Return
        A tuple containing the name of the .dat file that should be used to access each sample blocks, the start and end sample number, and n_samples of each block.
        """
        def get_file_and_start_index_within_file(files_first_sample,files_last_sample,start_index):
            file_index = [i for i in range(len(files_first_sample)) if start_index >= int(files_first_sample[i]) and start_index < int(files_last_sample[i])]
            return file_index
        
        my_block = []
        
        
        all_start = np.array([self.sample_index_per_channel[i][1] for i in range(len(self.sample_index_per_channel))])
        all_end = np.array([self.sample_index_per_channel[i][2] for i in range(len(self.sample_index_per_channel))])
        time = ([0, 6075360, 5875359], [55000, 6130360, 6130360])
        
        for t in range(0,len(time[0])):
            start = time[0][t]
            end = time[1][t]

            f1= get_file_and_start_index_within_file(files_first_sample = all_start, files_last_sample = all_end, start_index = start)
            f2= get_file_and_start_index_within_file(files_first_sample = all_start, files_last_sample = all_end, start_index = end)

            if f1 == f2:
                my_block.append((self.file_names[int(f1[0])], start, end, end-start))

            else:

                #

                for i in range(int(f1[0]),int(f2[0])+1): # loop through the .dat files
                #for i in [f1, f2]: # loop through the .dat files
                    if i == int(f1[0]): # first file

                        my_block.append((self.file_names[i], start, int(all_end[i]), int(all_end[i])-start))

                    elif i == int(f2[0]): # last file
                        my_block.append((self.file_names[i], int(all_start[i]), end, end-int(all_start[i])))


                    else: # files in the middle
                        my_block.append(self.sample_index_per_channel[i])
            return my_block