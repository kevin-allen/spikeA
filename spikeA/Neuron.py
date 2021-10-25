import numpy as np
from spikeA.Spike_train import Spike_train
from spikeA.Animal_pose import Animal_pose
from spikeA.Session import Session
from spikeA.Spatial_properties import Spatial_properties
from spikeA.Dat_file_reader import Dat_file_reader
from spikeA.Spike_waveform import Spike_waveform

class Neuron:
    """
    Class containing information about a single neuron.
    
    Attributes:
        name: Name of the session. Usually used as the beginning of the file names. Format is assumed to be subject-date-time
        subject: Name of the subject (animal).
        brain_area: Brain area in which the neuron was recorded
        channels: Channels on which the neuron was recorded. This is used to get the spike waveforms.
        spike_train: Spike_train object for the neuron.
        spike_waveform: Spike waveform object for the neuron.
        spatial_properties: Spatial_prop object for the neuron. Contains the single-cell spatial properties of the neuron
    Methods:
        set_spatial_properties()
        set_spike_train()
        
    """
    def __init__(self,name, subject=None, brain_area=None, channels=None,electrode_id=None):
        """
        Constructor of the Neuron Class
        """
        self.name = name
        self.subject = subject
        self.brain_area = brain_area
        self.channels = channels
        self.electrode_id = electrode_id
        
        # 3 types of analysis for a neurons (spike train, spatial properties and spike waveforms)
        # each will have its own class with associated attributes and methods
        self.spike_train = None
        self.spatial_properties = None
        self.spike_waveform = None
        return
    
    def set_spatial_properties(self, animal_pose):
        """
        Method of the neuron class to set the Spatial_properties object of the neuron
        
        Arguments
        animal_pose: Animal_pose object
        
        Return
        The neuron.spatial_properties object of the Neuron is set.
        """
        
        # if we don't have a Spike_train object in the Neuron object, create it
        if self.spike_train is None: # if None create the Spike_train object of the neuron
            raise TypeError("Set the neuron's Spike_train object before calling set_spatial_properties")
        
        #if not isinstance(animal_pose,Animal_pose): 
        #    raise TypeError("animal_pose should be a subclass of the Animal_pose class but is {}".format(type(animal_pose)))
        
        self.spatial_properties = Spatial_properties(animal_pose=animal_pose,spike_train=self.spike_train)
    
    
    def set_spike_train(self, sampling_rate = 20000, st = None):
        """
        Method of the neuron class to set the spike train of the neuron
        
        Arguments
        sampling_rate: sampling rate of the spike train
        st: 1D numpy array containing spike times of the neuron in seconds
        
        Return
        The neuron.spike_train object of the Neuron is set.
        """
        
        # if we don't have a Spike_train object in the Neuron object, create it
        if self.spike_train is None: # if None create the Spike_train object of the neuron
            self.spike_train = Spike_train(name=self.name,sampling_rate=20000)
        
                                           
        # We should normally have a valid st but we check just in case
        if st is None:
            raise TypeError("st should not be none")
        if not isinstance(st, np.ndarray):
            raise TypeError("st argument should be a numpy.ndarray but was {}".format(type(st)))
        if st.ndim != 1: # it should be one neuron so one dimention
            raise ValueError("st arguemnt should be a numpy array with 1 dimension but had {}".format(st.ndim))
                                      
        # call the set_spike_train of the Spike_train object of the neuron
        self.spike_train.set_spike_train(st=st)
    
    def set_spike_waveform(self,session=None,dat_file_reader=None):
        """
        set the Spike_waveform object of the Neuron
        
        
        """
        
        if self.spike_train is None:
            raise TypeError("self.spike_train.st should not be None")
        if session.n_channels is None:
            raise TypeError("ses.n_channels is None, run ses.load_parameter_files()")
        if not isinstance(session, Session):
            raise TypeError("session is not an instance of the Session class")
        
        if dat_file_reader is None:
            dat_file_reader= Dat_file_reader(session.dat_file_names,session.n_channels)
        else:
            if not isinstance(dat_file_reader,Dat_file_reader): 
                raise TypeError("dat_file is not an instance of Dat_file_reader class")
            
        self.spike_waveform = Spike_waveform(session = session, dat_file_reader=dat_file_reader, spike_train=self.spike_train)
        
        