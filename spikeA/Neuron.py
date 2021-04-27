import numpy as np
from spikeA.Spike_train import Spike_train

class Neuron:
    """
    Class containing information about a single neuron.
    
    Attributes:
        name: Name of the session. Usually used as the beginning of the file names. Format is assumed to be subject-date-time
        subject: Name of the subject (animal).
        brain_area: Brain area in which the neuron was recorded
        channels: Channels on which the neuron was recorded. This is used to get the spike waveforms.
        spike_train: Spike_train object for the neuron
        spike_waveform: Spike waveform object for the neuron
        spatial_prop: Spatial_prop object for the neuron. Contains the single-cell spatial properties of the neuron
    Methods:
        set_spike_train()
        
    """
    def __init__(self,name, subject, brain_area, channels):
        """
        Constructor of the Neuron Class
        """
        self.name = name
        self.subject = subject
        self.brain_area = brain_area
        self.channels = channels
        self.spike_train = None
        self.spatial_prop = None
        self.spike_waveform = None
        return
    
    def set_spike_train(sampling_rate = 20000, st = None):
        """
        Method of the neuron class to set the spike train of the neuron
        
        Arguments
        sampling_rate: sampling rate of the spike train
        st: spike times of the neuron in seconds
        
        Return
        None. The Spike_train object of the Neuron is set.
        """
        
        if self.spike_train is None: # if None create the Spike_train object of the neuron
            self.spike_train = Spike_train(name=self.name,sampling_rate=20000
                                           
        # We should normally have a valid st but we check just in case
        if st is not None:
            self.spike_train.set_spike_train(st=st)
        