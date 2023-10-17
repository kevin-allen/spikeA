import numpy as np
import spikeA
from spikeA.Spike_train import Spike_train
from spikeA.Animal_pose import Animal_pose
from spikeA.Session import Session
from spikeA.Session import Kilosort_session
from spikeA.Spatial_properties import Spatial_properties
from spikeA.Dat_file_reader import Dat_file_reader
from spikeA.Spike_waveform import Spike_waveform
#from spikeA.Session import Session #
from scipy.stats import multivariate_normal
from scipy.stats import poisson
from scipy.interpolate import interp1d
from scipy.stats import vonmises
import math


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
    def __init__(self, name, cluster_number=None, subject=None, brain_area=None, channels=None, shanks=None, electrode_id=None):
        """
        Constructor of the Neuron Class
        """
        self.name = name
        self.subject = subject
        self.brain_area = brain_area
        self.channels = channels
        self.shanks = shanks
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
        
        self.ap = animal_pose
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
        
        ##################################
        ### Rase value error depending on the class names 
        ##################################
        if not session.__class__ == spikeA.Session.Kilosort_session or session.__class__ == Session:
            raise TypeError("session is not an instance of the Session class")
            
        ###############
        ##### commented for now ## maryam 
        #########
        #if not isinstance(session, Session) or isinstance(session,spikeA.Session.Kilosort_session): ## or was added by maryam 
        #    raise TypeError("session is not an instance of the Session class")
        
        if dat_file_reader is None:
            dat_file_reader= Dat_file_reader(session.dat_file_names,session.n_channels)
        else:
            if not isinstance(dat_file_reader,Dat_file_reader): 
                raise TypeError("dat_file is not an instance of Dat_file_reader class")
            
        self.spike_waveform = Spike_waveform(session = session, dat_file_reader=dat_file_reader, spike_train=self.spike_train)
        
    
    def set_intervals(self, inter):
        """
        sets interval to both spike train of that neuron and to its spatial properties animal pose
        
        inter: Intervals
        """
        self.inter = inter
        self.spike_train.set_intervals(self.inter)
        self.ap.set_intervals(self.inter)
    
    def unset_intervals(self):
        """
        unsets interval of both spike train of that neuron and of its spatial properties animal pose
        """
        self.spike_train.unset_intervals()
        self.ap.unset_intervals()

        

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
    offset: np.array of shape (2,). Offset from 0,0 where the 3 component meets.
    orientation: np.array of shape (3,). Orientation in radian of the 3 components
    period: np.array of shape (3,). Period in cm for the 3 components. This is not the grid spacing.   period = grid spacing * np.cos(np.pi/6)
    peak_rate: firing field peak firing rate in Hz
    applyReLuToRate: whether or not we should pass the rate array into a relu function. See self.grid_cell_firing_rate() for more details. 
    
    sampling_rate: sampling rate of the spike train
    ap: Animal_pose object used to build the spike train
    
    """
    def __init__(self, name, 
                 offset=np.array([0,0]),
                 orientation = np.array([0.0,np.pi/3,np.pi/3*2]),
                 period = np.array([30,30,30]),
                 peak_rate=20, 
                 sampling_rate=20000, 
                 ap=None,
                 applyReLuToRate=True):
        super(Simulated_grid_cell,self).__init__(name=name)
    
        # variable defining a grid cell
        if offset.shape[0] != 2:
            raise ValueError("offset should be of shape (2,)")
        if orientation.shape[0] != 3:
            raise ValueError("orientation should be of shape (3,)")
        if period.shape[0] != 3:
            raise ValueError("period should be of shape (3,)")
            
        self.applyReLuToRate = applyReLuToRate
        self.offset=offset
        self.orientation=orientation
        self.period = period        
        self.peak_rate = peak_rate        
        
        #print("offset:",offset)
        #print("orientation:",orientation)
        
        
        self.sampling_rate = sampling_rate      
        self.ap = ap
        
        self.spike_train = Spike_train(name=self.name,sampling_rate=self.sampling_rate)
        
        self.remove_nan_from_ap()
        
        self.grid_cell_firing_rate()
        
        self.simulate_spike_train()
        
        self.spatial_properties = Spatial_properties(animal_pose=self.ap,spike_train=self.spike_train)
        
        self.spike_train.set_intervals(self.inter)
        self.ap.set_intervals(self.inter)
        
    def approxMolulo(self, x,maxValue):
        y = np.arctan(-1.0 / (np.tan(x/maxValue*np.pi))) + (0.5 * np.pi)
        y = y * maxValue/np.pi
        return 

   
    def pose_to_grid_cell_coordinate(self):
        """
        Function to transfrom the x,y position of the mouse to 
        a position within the internal representation of grid cells. 
    
        The internal representation is 3 angles (x,y,z) which represents the distance along 3 axes
        The 3 axes should be at approximately 60 degrees of each other.
        
        To get from distance to angle, we get the modulo of the distance and the underlying spacing.
        
        Set the angle in c0, c1, and c2. The range is -np.pi to pi. 
        
        """
                
        Rx0 = np.array([[np.cos(-self.orientation[0])],[-np.sin(-self.orientation[0])]]) # minus sign because we want to rotate the inverse of the angle to bring it back to 1,0 
        Rx1 = np.array([[np.cos(-self.orientation[1])],[-np.sin(-self.orientation[1])]])
        Rx2 = np.array([[np.cos(-self.orientation[2])],[-np.sin(-self.orientation[2])]])
                
        d0 = self.pose @ Rx0 # distance along axis 0
        d1 = self.pose @ Rx1
        d2 = self.pose @ Rx2

        self.c0 = (d0 % self.period[0])/self.period[0] * np.pi*2 # get angle from distance along axis. The new range is from 0 to 2*pi
        self.c1 = (d1 % self.period[1])/self.period[1] * np.pi*2 
        self.c2 = (d2 % self.period[2])/self.period[2] * np.pi*2 

        # set range to -np.pi to np.pi
        self.c0 = np.arctan2(np.sin(self.c0),np.cos(self.c0)) 
        self.c1 = np.arctan2(np.sin(self.c1),np.cos(self.c1))
        self.c2 = np.arctan2(np.sin(self.c2),np.cos(self.c2))
        
      
        
    def simulate_spike_train(self):
        """
        Get a spike train that is based on the spatial selectivity of the neuron
        """
        
        # higher sampling rate
        newTime = np.arange(start=self.ap.pose[0,0], stop = self.inter[0,1]-1,step=1/self.sampling_rate)
        
        # to interpolate the rate at a higher sampling rate
        fx = interp1d(self.ap.pose[:,0], self.rate) 
        
        mu = fx(newTime) # interpolate the rate 
        
        self.spike_train.generate_poisson_spike_train_from_rate_vector(mu ,sampling_rate=self.sampling_rate)
    
    
    def remove_nan_from_ap(self):
        """
        Remove the nan from the ap.pose.
        Only x,y and hd values are considered
        
        The ap object will be permenantly modified.
        If ap does not have np.nan value, this should not have any effect.
        """
        
        #print("Removing invalid values from ap")
        tStepSize = self.ap.pose[1,0]-self.ap.pose[0,0]
        pose = np.stack([self.ap.pose[:,0],self.ap.pose[:,1],self.ap.pose[:,2],self.ap.pose[:,4]]).T # only consider the data that we will be using
        keepIndices = ~np.isnan(pose).any(axis=1)
        maxT = np.sum(keepIndices)*tStepSize+tStepSize/2
        
        self.ap.pose = self.ap.pose[keepIndices]
        
        self.inter = np.array([[0,maxT]])
        
        self.ap.pose[:,0] = np.arange(start=tStepSize,stop = maxT,step=tStepSize)
        self.ap.pose_ori = self.ap.pose
        
    def relu_rate(self):
        """
        ReLU function to ensure there are no negative values in x. Negative numbers are set to 0.
        """
        self.rate = self.rate * (self.rate > 0) 
         
    def grid_cell_firing_rate(self):
        """
        Get the firing rate of grid cells
        """
        
        # we get the rate for all position
        self.pose = np.stack([self.ap.pose[:,1],self.ap.pose[:,2]]).T
        self.pose_to_grid_cell_coordinate() # now have self.c0, self.c1, self.c2
         

        self.offset = np.expand_dims(self.offset,0) # shift in cm
        # now we need to know the projection of the vector onto the 3 components
        Rx0 = np.array([[np.cos(-self.orientation[0])],[-np.sin(-self.orientation[0])]]) # minus sign because we want to rotate the inverse of the angle to bring it back to 1,0 
        Rx1 = np.array([[np.cos(-self.orientation[1])],[-np.sin(-self.orientation[1])]])
        Rx2 = np.array([[np.cos(-self.orientation[2])],[-np.sin(-self.orientation[2])]])
        d0 = self.offset @ Rx0
        d1 = self.offset @ Rx1
        d2 = self.offset @ Rx2
        self.phase = np.squeeze(np.array([(d0 % self.period[0])/self.period[0] * np.pi*2,
                               (d1 % self.period[1])/self.period[1] * np.pi*2,
                               (d2 % self.period[2])/self.period[2] * np.pi*2]))
        # from -np.pi to np.pi
        self.phase = np.arctan2(np.sin(self.phase),np.cos(self.phase))
            
        self.rateC0 = np.cos(self.c0-self.phase[0])
        self.rateC1 = np.cos(self.c1-self.phase[1])
        self.rateC2 = np.cos(self.c2-self.phase[2])
        
        # The sum of 3 components ranges from -1.5 to 3.0 when the angles are at 60 degrees of each other and the 3 axis have the same period. In this case, the (c0+c1+c2+1.5)/4.5 gives a range of 0 to 1.
        # If the axes are not at multiple of 60 degrees or periods are not equal, then the range is -3.0 to 3.0, the (c0+c1+c2+1.5)/4.5 gives a range from -1.5/4.5 to 1.
        # The consequence is that for axes that are not perfectly at 60 degrees of each other or where the 3 periods are not equal, we get some negative firing rate values.
        # We can apply a ReLu function to get rid of the negative data. This is a differentiable function.
        
        self.rate =  np.squeeze(((self.rateC0+self.rateC1+self.rateC2+1.5)/4.5*self.peak_rate))
        
        if self.applyReLuToRate:
            self.relu_rate()
        
    
        
#Simulate a HD cell
class Simulated_HD_Cell(Neuron):
    '''
    Simulate a HD cell based on the von Mises distribution. https://en.wikipedia.org/wiki/Von_Mises_distribution
    
    Arguments
    name: Name of stimulated HD cell
    peakAngle: Peak position of the HD cell in radian
    sharpness: How sharply tunned the HD cell is, the larger the number the sharper
    ap: Animal position used to build the spike train
    peakRate: Peak firing rate
    sampling_rate: sampling rate of the spike train, default 20000
    
    Usage:
    Same as the Neuron class
    
    Examples:
    testHDNeuron = Simulated_HD_Cell('testHD',ap = sSes.ap,peakAngle = 90,sharpness=5,peakRate=20)
    
    testHDNeuron.spatial_properties.firing_rate_head_direction_histogram(deg_per_bin=10, smoothing_sigma_deg=10,smoothing=True)  
    
    testNeuronHistos = testHDNeuron.spatial_properties.firing_rate_head_direction_histo
    '''
    
    def __init__(self,name,peakAngle=0,sharpness=1,peakRate = 10,sampling_rate = 20000, ap=None):        
        super().__init__(name=name)
        
        self.name = name
        self.peakAngle = math.radians(peakAngle)
        self.sharpness = sharpness
        self.ap = ap
        self.peakRate = peakRate
        self.sampling_rate = sampling_rate
        self.spike_train = Spike_train(name=self.name,sampling_rate=self.sampling_rate)
        
        self.remove_nan_from_ap()
        self.simulate_spike_train()
        
        self.spatial_properties = Spatial_properties(animal_pose=self.ap,spike_train=self.spike_train)
        
        self.spike_train.set_intervals(self.inter)
        self.ap.set_intervals(self.inter)
        
    def simulate_spike_train(self):
        newTime = np.arange(start=self.ap.pose[0,0], stop = self.inter[0,1]-1,step=1/self.sampling_rate)
        
        fHD = interp1d(self.ap.pose[:,0], self.ap.pose[:,4]) # create function that will interpolate the Head Direction
        
        hdNew = fHD(newTime) # interpolate from new time
        
        #Get the rate at each sampling data point using the vonmises probability density function
        mu = vonmises.pdf(x = hdNew, loc = self.peakAngle, kappa = self.sharpness,scale = 1) * self.peakRate 
        
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