import numpy as np
from spikeA.Spike_train import Spike_train
from spikeA.Animal_pose import Animal_pose
from spikeA.Session import Session
from spikeA.Spatial_properties import Spatial_properties
from spikeA.Dat_file_reader import Dat_file_reader
from spikeA.Spike_waveform import Spike_waveform

from scipy.stats import multivariate_normal
from scipy.stats import poisson
from scipy.interpolate import interp1d

class Neuron:
    """
    Class containing information about a single neuron.
    
    Attributes:
        name: Name of the neuron. Usually used as the beginning of the file names and ending with the clustering number. Format is assumed to be subject-date-time
        cluster_number: Number obtained from the Spike_train_loader, usually from the clustering procedure.
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
    def __init__(self, name, cluster_number=None, subject=None, brain_area=None, channels=None, electrode_id=None):
        """
        Constructor of the Neuron Class
        """
        self.name = name
        self.subject = subject
        self.brain_area = brain_area
        self.channels = channels
        self.electrode_id = electrode_id
        
        self.cluster_number = cluster_number
        
        # 3 types of analysis for a neurons (spike train, spatial properties and spike waveforms)
        # each will have its own class with associated attributes and methods
        self.spike_train = None
        self.spatial_properties = None
        self.spike_waveform = None
    
    def set_spatial_properties(self, animal_pose):
        """
        Method of the neuron class to set the Spatial_properties object of the neuron
        The Neuron.spatial_properties.spike_train will be set to its Neuron.spike_train for each neuron
        
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
        

class Simulated_place_cell(Neuron):
    """
    Class to simulate the spike train of a place cell.
    
    The firing rate in 2D space is simulated with a 2D gaussian kernel
    The neuron has 1 firing field
    
    Arguments
    name: name of the simulated place cell
    peak_lock: location of the firing field peak
    standard_deviation: std of the firing field
    peak_rate: firing field peak firing rate in Hz
    sampling_rate: sampling rate of the spike train
    ap: Animal_pose object used to build the spike train
    
    """
    def __init__(self, name, peak_loc=[0,0], standard_deviation=10, peak_rate=20, sampling_rate=20000, ap=None):
        super(Simulated_place_cell,self).__init__(name=name)
        
        self.peak_loc=peak_loc
        self.standard_deviation=standard_deviation
        self.peak_rate = peak_rate
        self.sampling_rate = sampling_rate      
        self.ap = ap
        self.spike_train = Spike_train(name=self.name,sampling_rate=self.sampling_rate)
        
        self.remove_nan_from_ap()
        self.simulate_spike_train()
        
        self.spatial_properties = Spatial_properties(animal_pose=self.ap,spike_train=self.spike_train)
        
        self.spike_train.set_intervals(self.inter)
        self.ap.set_intervals(self.inter)
        

    def simulate_spike_train(self):
        """
        Get a spike train that is based on the spatial selectivity of the neuron
        """
        newTime = np.arange(start=self.ap.pose[0,0], stop = self.inter[0,1]-1,step=1/self.sampling_rate)
        
        fx = interp1d(self.ap.pose[:,0], self.ap.pose[:,1]) # create function that will interpolate
        fy = interp1d(self.ap.pose[:,0], self.ap.pose[:,2]) # create function that will interpolate

        xNew = fx(newTime) # interpolate
        yNew = fy(newTime) # interpolate
        
        xy = np.stack([xNew,yNew]).T
        
        # get the rate at each sampling data point
        mu = multivariate_normal.pdf(x=xy,mean = self.peak_loc, cov = [[self.standard_deviation**2, 0], [0, self.standard_deviation**2]])* (2*np.pi *self.standard_deviation**2) * self.peak_rate
        
        self.spike_train.generate_poisson_spike_train_from_rate_vector(mu ,sampling_rate=self.sampling_rate)
        
    def remove_nan_from_ap(self):
        """
        Remove the nan from the ap.pose.
        Only x,y and hd values are considered
        
        The ap object will be permenantly modified.
        If ap does not have np.nan value, this should not have any effect.
        """
        
        
        tStepSize = self.ap.pose[1,0]-self.ap.pose[0,0]
        pose = np.stack([self.ap.pose[:,0],self.ap.pose[:,1],self.ap.pose[:,2],self.ap.pose[:,4]]).T # only consider the data that we will be using
        keepIndices = ~np.isnan(pose).any(axis=1)
        maxT = np.sum(keepIndices)*tStepSize+tStepSize/2
        
        self.ap.pose = self.ap.pose[keepIndices]
        
        self.inter = np.array([[0,maxT]])
        
        self.ap.pose[:,0] = np.arange(start=tStepSize,stop = maxT,step=tStepSize)
        self.ap.pose_ori = self.ap.pose

        
class Simulated_grid_cell(Neuron):
    """
    Class to simulate the spike train of a grid cell.
    
    The firing rate in 2D space is simulated with cos function
    
    Arguments
    name: name of the simulated grid cell
    offset: np.array of shape 2,1 or 2,
    orientation: in radian
    spacing: distance between closest firing fields
    peak_rate: firing field peak firing rate in Hz
    
    sampling_rate: sampling rate of the spike train
    ap: Animal_pose object used to build the spike train
    
    """
    def __init__(self, name, 
                 offset=np.array([[0],[0]]),
                 orientation = 0.0,
                 spacing = 30,
                 peak_rate=20, 
                 sampling_rate=20000, 
                 ap=None):
        super(Simulated_grid_cell,self).__init__(name=name)
    
        # variable defining a grid cell
        if offset.ndim ==1:
            offset = np.expand_dims(offset,1)
        
        
        self.offset=offset
        self.orientation=orientation
        self.spacing = spacing
        self.peak_rate = peak_rate
        
        
        self.sampling_rate = sampling_rate      
        self.ap = ap
        self.spike_train = Spike_train(name=self.name,sampling_rate=self.sampling_rate)
        
        self.remove_nan_from_ap()
        self.simulate_spike_train()
        
        self.spatial_properties = Spatial_properties(animal_pose=self.ap,spike_train=self.spike_train)
        
        self.spike_train.set_intervals(self.inter)
        self.ap.set_intervals(self.inter)
        

    def simulate_spike_train(self):
        """
        Get a spike train that is based on the spatial selectivity of the neuron
        """
        
        # we get the rate for all position
        poses = np.stack([self.ap.pose[:,1],self.ap.pose[:,2]]).T
        
        deltas = [self.orientation, self.orientation+np.pi/3, self.orientation+np.pi/3*2]
        
        rates = np.apply_along_axis(self.grid_firing_rate_at_p,1,
                                   poses, self.offset,deltas,self.spacing,self.peak_rate)
        
        # higher sampling rate
        newTime = np.arange(start=self.ap.pose[0,0], stop = self.inter[0,1]-1,step=1/self.sampling_rate)
        
        # to interpolate the rate at a higher sampling rate
        fx = interp1d(self.ap.pose[:,0], rates) 
        

        mu = fx(newTime) # interpolate the rate 
        
        self.spike_train.generate_poisson_spike_train_from_rate_vector(mu ,sampling_rate=self.sampling_rate)
    
    
    def grid_firing_rate_at_p(self,pose,offset,deltas,spacing,peak_rate):
        """
        Function to calculate the firing rate of a grid cell at any 2D position

        Arguments
        pose: np.array of shape (2,1)
        offset: np.array of shape (2,1)

        """
        if pose.ndim == 1:
            pose = np.expand_dims(pose,1)
    
        p = pose-offset # I change this from + to -

        # we want to get the x value of the position vector, after rotating the position vector by different amount 
        # these are like rotation matrices, but we remove the terms that would give us the y component. 
        Rx0 = np.array([[np.cos(deltas[0]),-np.sin(deltas[0])]])
        Rx1 = np.array([[np.cos(deltas[1]),-np.sin(deltas[1])]])
        Rx2 = np.array([[np.cos(deltas[2]),-np.sin(deltas[2])]])

        c0 = np.cos(Rx0@p * (np.pi*2) / spacing) # we rotate the position vector and only get the x value of the rotated vector, than we normalize so that there is one cycle per spacing, cos function gives the oscillation
        c1 = np.cos(Rx1@p * (np.pi*2) / spacing)
        c2 = np.cos(Rx2@p * (np.pi*2) / spacing)


        return (((c0+c1+c2)+1.5)/(3.0+1.5) * peak_rate)[0,0] # we sum the 3 components,  normalize so that the range is from 0 to 1, then multiply by the value that we want as peak rate

    
    
    
    def remove_nan_from_ap(self):
        """
        Remove the nan from the ap.pose.
        Only x,y and hd values are considered
        
        The ap object will be permenantly modified.
        If ap does not have np.nan value, this should not have any effect.
        """
        
        
        tStepSize = self.ap.pose[1,0]-self.ap.pose[0,0]
        pose = np.stack([self.ap.pose[:,0],self.ap.pose[:,1],self.ap.pose[:,2],self.ap.pose[:,4]]).T # only consider the data that we will be using
        keepIndices = ~np.isnan(pose).any(axis=1)
        maxT = np.sum(keepIndices)*tStepSize+tStepSize/2
        
        self.ap.pose = self.ap.pose[keepIndices]
        
        self.inter = np.array([[0,maxT]])
        
        self.ap.pose[:,0] = np.arange(start=tStepSize,stop = maxT,step=tStepSize)
        self.ap.pose_ori = self.ap.pose
        