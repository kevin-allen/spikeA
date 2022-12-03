"""
File containing the definition of the Dat_file_reader class
"""
import numpy as np
import pandas as pd
import os
import os.path
import pickle
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
    total_samples
    files_first_sample
    files_last_sample
    
    Methods:
    __get_first_last_samples_each_file()
    get_data_one_block()
    __get_block_start_end_within_files()
    __read_one_block()

    """
    def __init__(self,file_names,n_channels): 
        """
        Constructor of the Dat_file_reader class

        Arguments:
        file_names: List containing the full path of the .dat files
        n_channels: number of channels in the files
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
        
        # total number of samples
        self.total_samples=np.sum(self.samples_per_file)
        
        
        # get the first and last sample of each file
        self.files_first_sample, self.files_last_sample = self.get_first_last_samples_each_file()
        
        # variables set when loading a chunk of .dat data into RAM
        self.start_sample_in_ram=None
        self.max_sample_in_ram=None
        self.ram_block_loaded= False 
        
    
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    
    def read_data_to_ram(self,read_size_gb=5):
        """
        Function to read a large chunk of data from dat files and keep it in RAM
        It will contain all channels contained in the .dat files
        Call self.release_ram() when done with data to prevent filling your RAM 
        """
        if(read_size_gb>20):
            print("You just asked the Dat_file_reader to load more than 20 Gb into RAM")
        
        self.max_read_size=read_size_gb*1000000000
        self.start_sample_in_ram=0
        self.max_sample_in_ram=int(self.max_read_size/(self.n_channels*2))
        if self.max_sample_in_ram > self.total_samples:
            self.max_sample_in_ram = self.total_samples
        print("loading {} GB of data (sample {} to {}) to RAM".format(read_size_gb,self.start_sample_in_ram,self.max_sample_in_ram))
        self.RAM_block = self.get_data_one_block(self.start_sample_in_ram,self.max_sample_in_ram,np.arange(self.n_channels))
        self.ram_block_loaded= True 
    def release_ram(self):
        """
        
        Function to free the memory block that was used by the .dat files
        """
        self.start_sample_in_ram=None
        self.max_sample_in_ram=None
        self.RAM_block=None
        self.ram_block_loaded = False
    
    def get_first_last_samples_each_file(self):
        """
        Calculate what the first and last sample for the dat files.

        Arguments:

        Returns: 
        tuple containing 2 1D numpy arrays (first and last sample in each file)
        """
        files_last_sample = np.cumsum(self.samples_per_file)-1
        files_first_sample = np.insert(files_last_sample+1,0,0)[0:-1]
        return files_first_sample, files_last_sample

    def get_file_intervals_in_seconds(self,sampling_rate=20000):
        """
        Calculate the start and end time of dat files in seconds
        
        Argument:
        sampling_rate: sampling rate (Hz) to calculate time in seconds, default 20000
        
        Returns:
        2D numpy array with one interval per row
        """
        s,e = self.get_first_last_samples_each_file()
        return np.stack([s/sampling_rate,e/sampling_rate],axis=1)
    
    def get_file_duration_seconds(self,sampling_rate=20000):
        """
        Calculate the duration in second of each .dat files
        
        Argument:
        sampling_rate: sampling rate (Hz) to calculate time in seconds, default 20000
        
        Returns:
        1D numpy array with the duration in seconds for each .dat files
        """
        
        s,e = self.get_first_last_samples_each_file()
        return (e-s)/sampling_rate
    
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
        
        if not isinstance(start_sample,(int, np.integer)):
            raise ValueError("start_sample should be an integer but is {}".format(type(start_sample)))
        if not isinstance(end_sample,(int,np.integer)):
            raise ValueError("end_sample should be an integer but is {}".format(type(end_sample)))
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

        if self.ram_block_loaded and start_sample > self.start_sample_in_ram and end_sample < self.max_sample_in_ram:
            # get from RAM data
            return self.RAM_block[channels,start_sample:end_sample]
            
        else:
            # get from dat files
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
        A tuple with start_file_no, start_index_within_file, end_file_no, end_index_within_file
        """
        
        # check that arguments make sense
        if start_index < 0:
            raise ValueError("start_index should be 0 or larger")
        if end_index <= start_index:
            raise ValueError("end_index should be larger than start_index")
        if end_index > self.total_samples:
            raise ValueError("end_index is larger than the number of samples in the Dat_file_reader object")
        
        # get the starting point of reading operation in dat files (start_file_no,start_index_within_file)
        start_file_no = np.where((start_index >=self.files_first_sample) & (start_index <self.files_last_sample))[0].item()
        start_index_within_file = int(start_index - self.files_first_sample[start_file_no])
        
        # get the ending point of reading operation in dat files
        end_file_no = np.where((end_index >=self.files_first_sample) &  (end_index <= self.files_last_sample))[0].item()
        end_index_within_file = int(end_index - self.files_first_sample[end_file_no])
       
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
            #print("can read the block from a single file")
            #print("Read file: ",self.file_names[f1], " from ", i1, "to" , i2)
            my_mapped_file = np.memmap(self.file_names[f1], dtype = "int16", mode = "r",
                                       shape = (self.n_channels, self.samples_per_file[f1]), order = "F")                    
            my_block = my_mapped_file[channels,i1:i2]
        else:
            #print ("read the block from several files")
            # allocate the memory for the whole block
            my_block = np.empty((len(channels),samples_to_read),dtype="int16") # something similar
            copied = 0
            for i in range(f1,f2+1): # loop through the .dat files we need
                #print("reading from file ",i)
                my_mapped_file = np.memmap(self.file_names[i], dtype = "int16", mode = "r",
                                       shape = (self.n_channels, self.samples_per_file[i]), order = "F")                       
                if i == f1: # first file    
                    #print("copy from ",i1," to the end of the file (",self.samples_per_file[i],") in file ",f1)
                    #print("into 0 to ", self.samples_per_file[i]-i1)
                    #print(self.samples_per_file[i]-i1)
                    my_block[np.arange(len(channels)), 0:(self.samples_per_file[i]-i1)] = my_mapped_file[channels,i1:self.samples_per_file[i]]
                    copied = copied + self.samples_per_file[i]-i1

                elif i == f2: # last file
                    #print("copy from the begining of the file ",f2, " to ", i2)
                    #print("copied:",copied, " of ", samples_to_read)
                    my_block[np.arange(len(channels)), copied:] = my_mapped_file[channels,0:i2]

                else: # files in the middle
                    #print("read the entire file ",i)
                    my_block[np.arange(len(channels)), copied:(copied+self.samples_per_file[i])] = my_mapped_file[channels,:]
                    copied = copied + self.samples_per_file[i] 
        return my_block
    
    def merge_trial_dat_files(self,outFileName):
        """
        Create a single .dat file from the list of .dat files in the Dat_file_reader object.
        
        Under the hood it uses the `cat` command to concatenate the files
        
        The function also creates a pickle of the Dat_file_reader (self) so that the user can recreate the single .dat files from the merged .dat file.
        
        Argument:
        outputFileName: path of the output file (single .dat file)
        """
       
        
        pickle_fn = os.path.splitext(outFileName)[0]+"_Dat_file_reader.pickle"
        print("Create pickle of the Dat_file_reader: " + pickle_fn)
        filehandler = open(pickle_fn, 'wb') 
        pickle.dump(self, filehandler)
    
        
        print("Merged .dat file size:{:.2f} Gb".format(np.sum(self.size_of_files)/1000000000))
        print("Create " + outFileName)
        com="cat "+ ' '.join(self.file_names)+" > "+outFileName
        print("running: " , com)
        os.system(com)
        
        # check the size of the file againts the sum of individual files
        merge_size = os.path.getsize(outFileName)
        if merge_size != np.sum(self.size_of_files):
            raise ValueError("Problem with size of merged file {}, {},{}".format(outFileName,merge_size,np.sum(self.size_of_files)))
        else:
            print("File sizes match {}, {},{}".format(outFileName,merge_size,np.sum(self.size_of_files)))
            
            
    def delete_merged_dat_files(self,path):

        """
        This function checks if the merged dat file exists. If yes, it checks if all single dat files exist and if the waveforms have been created. If yes, it removes the merged dat file. 
        
        Argument:
        path to the session (eg "/adata/electro/mouse/mouse-date-01xx")
        """
        
        name = str(path).split("/")[-1]
        desen = open(f"{str(path)}/{name}.desen").read().split('\n')[:-1]
        mouse_name=name.split('-')[0]
        date_name=name.split('-')[1]
        number_of_trials = len(desen)

        #check if the merged dat file still exists
        merged_dat=f"{path}/{name}.dat"
        if not os.path.exists(merged_dat):
            print(f"{merged_dat} has already been deleted.")
        else:
            #check that all single dat files exist
            dat_files=[f"{path}/{mouse_name}-{date_name}_0{trial+1}.dat" for trial in range(number_of_trials)]
            for dat_file in dat_files:
                if not os.path.exists(dat_file):
                    raise ValueError(f"{dat_file} is missing")

            #check that the waveforms have been created
            file=f"{path}/{name}.waveforms.npy"
            if not os.path.exists(file):
                print("Creating waveforms")
                Spike_waveform.save_waveforms(path=path)
                #raise ValueError(f"For {name}, you need to create the waveforms first")
                
            else:
                os.remove(merged_dat)
                print(f"Deleted {merged_dat}")
