import numpy as np
from spikeA.Animal_pose import Animal_pose
from spikeA.Spike_train import Spike_train
from scipy.interpolate import interp1d
from spikeA.Dat_file_reader import Dat_file_reader
from spikeA.Session import *


class Spike_waveform:
    """
    Class use to calculate the spike waveform of a single neuron.
    
    Attributes:
        st = Spike_train object
        df = Dat_file_reader object
        ses = Session object
    Methods:
        mean_wave_form()
        
    """
    def __init__(self, Session = None, dat_file=None, spike_train=None):
        """
        Constructor of the Spike_waveform class
        """
        
#         if not isinstance(spike_train,Spike_train): 
#             raise TypeError("spike_train should be a subclass of the Spike_train class")
#         if not isinstance(dat_file,Dat_file_reader): 
#             raise TypeError("animal_pose should be a subclass of the Dat_file_reader class")
        
        self.ses = Session    
        self.st = spike_train
        self.df = dat_file
        
        return
    
    def mean_waveform(self, block_size, channels):
        """
        A method to get the mean waveform of all neurons
        """
    
        blocks = np.ndarray((len(channels), block_size, self.st.n_spikes()))
        spike_time_sample = self.st.st * self.st.sampling_rate
        
        my_list_of_tuples = [self.df.get_block_start_end_within_files(s-block_size/2, s+block_size/2) for s in spike_time_sample]
        
        bl=0
        
        for f1,i1,f2,i2 in my_list_of_tuples :
            blocks[:,:,bl] = self.df.read_one_block(f1,int(i1),f2,int(i2),block_size,channels)
            bl=bl+1
            
        self.spike_waveform = blocks
        self.mean_waveforms =  np.mean(blocks, axis = 2)
        
 


