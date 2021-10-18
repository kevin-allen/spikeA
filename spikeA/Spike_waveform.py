import numpy as np
from spikeA.Animal_pose import Animal_pose
from spikeA.Spike_train import Spike_train
from scipy.interpolate import interp1d
from spikeA.Dat_file_reader import Dat_file_reader
from spikeA.Session import Session



class Spike_waveform:
    """
    Class use to calculate the spike waveform of a single neuron.
    
    Attributes:
        ses = Session object
        st = Spike_train object
        dfr = Dat_file_reader object
        
    Methods:
        mean_wave_form()
        
    """
    def __init__(self, session = None, dat_file_reader=None, spike_train=None):
        """
        Constructor of the Spike_waveform class
        """
        if not isinstance(session, Session):
            raise TypeError("session is not an instance of the Session class")
        if not isinstance(spike_train,Spike_train): 
            raise TypeError("spike_train is not an instance of Spike_train class")
        if not isinstance(dat_file_reader,Dat_file_reader): 
            raise TypeError("dat_file is not an instance of Dat_file_reader class")
        
        self.ses = session    
        self.st = spike_train
        self.dfr = dat_file_reader
        
        return
    
    
    
    def mean_waveform(self,block_size, channels, n_spikes=None):
        """
        Method to get the mean waveform of one neuron on all channels
        
        It first gets all the waveforms in a 3D array and then get the mean to reduce the array to a 2D array [channels,block_size]
        
        Arguments:
        block_size = Number of time points in the waveform
        channels= channel list as 1D np.array
        n_spikes= if you set this to a positive integer, it will limit the analysis to the first n spikes. By default, the value is None and all spikes are analyzed
        
        Return:
        The function does not return anything but create self.spike_waveform and self.mean_waveforms
        self.spike_waveform is a 3D array [channels,block_size,spikes]
        self.mean_waveform is a 2D array [channels,block_size]
        """
        
        if n_spikes is not None:
            if n_spikes < 1:
                 raise ValueError("n_spikes should be a positive value")
            
        # Determine how many spikes will be considered 
        if n_spikes is None:
            n_spikes = self.st.n_spikes()
        else:
            if n_spikes > self.st.n_spikes(): # if n_spike is larger than number of spikes, set it to number of spikes
                n_spikes= self.st.n_spikes()
            
        # Create the blocks array to hold the spike waveform of the spikes in memory 
        blocks = np.ndarray((len(channels), block_size, n_spikes))
        
        # Transform spike times from from seconds to samples in .dat files
        spike_time_sample = np.round(self.st.st[:n_spikes] * self.st.sampling_rate)
        
        # Remove any spike window that would start before 0 or end after the file ends
        spike_time_samples=spike_time_sample[np.logical_and(spike_time_sample-block_size/2 > 0,spike_time_sample+block_size/2 < self.dfr.total_samples)]
        
        # Get the start and end of the reading operation for each spike
        read_block_limits = [self.dfr.get_block_start_end_within_files(s-block_size/2, s+block_size/2) for s in spike_time_sample]
        print("n_spikes: {}, read from file {} to file {}, shape: {}".format(n_spikes,read_block_limits[0][0],read_block_limits[-1][2],blocks.shape))
        
        bl=0
        for f1,i1,f2,i2 in read_block_limits :
            blocks[:,:,bl] = self.dfr.read_one_block(f1,i1,f2,i2,block_size,channels)
            bl=bl+1
            
        self.spike_waveform = blocks
        self.mean_waveforms =  np.mean(blocks, axis = 2)
        
    
    def largest_amplitude_waveform(self):
        """
        A function to get the largest amplitude waveform
        self.mean_waveforms is a 2D array with time and channel as dimentions
        here we find the channel with the largest amplitude and return the waveform associated to that.

        returns: the largest_amplitude waveform 
        """
        if self.mean_waveforms is None:
            raise ValueError("set_mean_waveforms should be set before running the largest_amplitude_waveform")

            
        self.largest_wf= self.mean_waveforms[np.argmax(np.ptp(self.mean_waveforms,axis=1)),:]
