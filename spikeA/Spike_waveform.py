import numpy as np
from spikeA.Animal_pose import Animal_pose
from spikeA.Spike_train import Spike_train
from scipy.interpolate import interp1d
from spikeA.Dat_file_reader import Dat_file_reader
from spikeA.Session import Session
from spikeA.Session import Kilosort_session
#from spikeA.Cell_group import Cell_group
from spikeA.Spike_train_loader import Spike_train_loader
from spikeA.Intervals import Intervals
import os

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
        if not (isinstance(session, Session) or isinstance(session, Kilosort_session)) :
            raise TypeError("session is not an instance of the Session class")
        if not isinstance(spike_train,Spike_train): 
            raise TypeError("spike_train is not an instance of Spike_train class")
        if not isinstance(dat_file_reader,Dat_file_reader): 
            raise TypeError("dat_file is not an instance of Dat_file_reader class")
        
        self.ses = session    
        self.st = spike_train
        self.dfr = dat_file_reader
        self.channels = None
        
        return
    
    
    
    def mean_waveform(self, channels, block_size=200, n_spikes=None):
        """
        Method to get the mean waveform of one neuron on all channels
        
        It first gets all the waveforms in a 3D array and then get the mean to reduce the array to a 2D array [channels,block_size]
        
        Arguments:
        block_size = Number of time points in the waveform (centered around spike)
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
        
        self.channels=channels
        
        # Create the blocks array to hold the spike waveform of the spikes in memory 
        blocks = np.empty((len(self.channels), block_size, n_spikes))
        
        # Transform spike times from from seconds to samples in .dat files
        spike_time_sample = np.round(self.st.st[:n_spikes] * self.st.sampling_rate)
        
        # Remove any spike window that would start before 0 or end after the file ends
        spike_time_sample = spike_time_sample[np.logical_and(spike_time_sample-block_size/2 > 0,spike_time_sample+block_size/2 < self.dfr.total_samples)]
        
        # get a block of data for each spike and save them in our 3D array
        bl=0        
        for t in spike_time_sample :
            blocks[:,:,bl] = self.dfr.get_data_one_block(int(t-block_size/2),int(t+block_size/2),self.channels)
            bl=bl+1
        
        # save all waveforms (3D array)
        self.spike_waveform = blocks
        # get and save the mean of all waveforms around spikes, results in a 2D array (calculate mean across last axis, that is for each n_spikes)
        self.mean_waveforms = np.mean(blocks, axis = 2)
        
    
    def largest_amplitude_waveform(self):
        """
        A function to get the largest amplitude waveform
        self.mean_waveforms is a 2D array with time and channel as dimentions
        here we find the channel with the largest amplitude and return the waveform associated to that.

        returns: the largest_amplitude waveform 
        """
        if self.mean_waveforms is None:
            raise ValueError("mean_waveform() should be run before running largest_amplitude_waveform()")

        max_index = np.argmax(np.ptp(self.mean_waveforms,axis=1))
        self.max_amplitude_channel = self.channels[max_index]
        self.largest_wf= self.mean_waveforms[max_index,:]
        
        
        
    def save_waveforms(self, path, block_size=200, overwrite=False, n_spikes=5000):
        """
        This function creates the mean waveform on each recording channel for each cell and saves it.

            returns: array with waveforms
        """    
        # load session
        name = path.rstrip("/").split("/")[-1].split("_")[0]
        ses = Kilosort_session(name=name,path=path)
        ses.load_parameters_from_files()
        stl = Spike_train_loader()
        stl.load_spike_train_kilosort(ses)
        cg = Cell_group(stl)

        stim_trials = [i for i, j in enumerate(ses.stimulation) if j !='none']
        if len(stim_trials):
            recording_channels=ses.n_channels-2
        else:
            recording_channels=ses.n_channels-1
        file=f"{path}/{name}.waveforms.npy"

        if not os.path.exists(file) or overwrite==True:
            print(f'creating waveforms for {name}')
            df = Dat_file_reader(file_names=[f"{path}/{name}.dat"], n_channels=ses.n_channels)

            #template for array:
            array = np.zeros((recording_channels, block_size, len(cg.neuron_list)))

            for i,n in enumerate(cg.neuron_list):
                print("i/n",i,n.name)
                sw = Spike_waveform(session=ses, dat_file_reader=df, spike_train=n.spike_train)
                sw.mean_waveform(block_size=block_size, channels=np.arange(recording_channels), n_spikes=n_spikes) #calculate the mean waveforms for all channels over a time interval of block_size
                array[:,:,i]=sw.mean_waveforms

            np.save(file = file , arr = array)
            return array
        else:
            print(f"waveforms for {name} have already been created")
            return np.load(f"{path}/{name}.waveforms.npy")
