import numpy as np
from itertools import permutations, combinations
from tqdm import tqdm
from spikeA.Neuron import Neuron
from spikeA.Spike_train_loader import Spike_train_loader

class Cell_group:
    """
    Class dealing with a group of neurons recorded in the same session.
    
    This class makes it easier to apply the same analysis on a list of neurons.
    The Cell_group class can also been used to do analysis of simultaneously recorded cells. 
    
    
    Attributes:
        neuron_list: List of Neuron objects
        pair_indices: List of size-2 tuples containing the index of neurons in the pairs.
    
    Methods:
        __init__
        make_pairs(): Get the indices of pairs of cells. 
    
    """
    
    def __init__(self,stl):
        """
        We create a list of Neuron object and set the spike trains of the neurons using a Spike_train_loader object. 
        
        Arguments:
        stl: Spike_train_loader object 
        
        # This function creates a list of Neurons and set the spike train object using the spike_train_loader object
        """
        if not isinstance(stl,Spike_train_loader):
            raise TypeError("stl should be a Spike_train_loader object but is {}".format(type(stl)))
        
        ## create a list of neurons
        ## use a list comprehension, use the stl.clu_ids to set the name of the neurons
        self.neuron_list=[Neuron(name=str(clu)) for clu in stl.clu_ids]
        
        ## set the spike_train objects of your neurons
        for i,n in enumerate(self.neuron_list):
               n.set_spike_train(st=stl.spike_times[i])
        
    def set_info_from_session(self, ses, maxchannels=5):
        """
        set Neuron information from session (see Neuron class for more information)
        consider the $maxchannels channels with highest amplitude
        """
        
        ses.load_waveforms()
        ses.init_shanks()
        
        for n in self.neuron_list:
            clu_id = int(n.name)
            channels = ses.get_channels_from_cluster(clu_id, maxchannels)
            shanks_arr, active_shanks, electrodes = ses.get_active_shanks(channels)
            n.channels = channels
            n.brain_area = electrodes
            # n.electrode_id
    
    def make_pairs(self,pair_type="permutations"):
        """
        Get the indices of neurons in pairs of neurons
        
        The type of pairs can be permutations or combinations. See the itertools.permutations and itertools.combinations function for more details.
        itertools.permutations returns a->b and b->a wherease itertools.combinations returns a->b only.
        
        Argument:
        pair_type: type of pairs you want to create
        
        Returns:
        self.pairs: A list of size-2 tuples in the Cell_group object.
        """
        if pair_type=="permutations":
            self.pairs = list(permutations(range(len(self.neuron_list)),2))
        if pair_type=="combinations":
            self.pairs = list(combinations(range(len(self.neuron_list)),2))
    
    def spike_time_crosscorrelation(self,bin_size_sec=0.0005, min_sec=-0.1, max_sec=0.1):
        """
        Calculate the spike-time crosscorrelation for all cell pairs in self.pairs.
        
        Arguments:
        bin_size_sec: size of bins in the histogram
        min_sec: minimum value in the histogram
        max_sec: maximum value in the histogram
        
        Returns:
        self.st_crosscorrelation: 2D np.ndarray with one spike time crosscorrelation per row
        """
        
        myRange = np.arange(min_sec,max_sec+bin_size_sec,bin_size_sec)
        self.st_crosscorrelation = np.ndarray((len(self.pairs),myRange.shape[0]-1)) # to store the results
        
        for i in tqdm(range(len(self.pairs))):
            j,k = self.pairs[i]
            self.st_crosscorrelation[i,:] = self.neuron_list[j].spike_train.spike_time_crosscorrelation(st2=self.neuron_list[k].spike_train.st,
                                                                                                bin_size_sec = bin_size_sec, 
                                                                                                min_sec = min_sec, max_sec = max_sec)[0] # [0] keeps only the counts
        self.st_crosscorrelation_bins = myRange
        
    def spike_time_autocorrelation(self,bin_size_sec=0.0005, min_sec=-0.1, max_sec=0.1):
        """
        Calculate the spike-time autocorrelation for all neurons in the Cell_group object.
        
        Arguments:
        bin_size_sec: size of bins in the histogram
        min_sec: minimum value in the histogram
        max_sec: maximum value in the histogram
        
        Returns:
        self.st_autocorrelation: 2D np.ndarray with one spike time autocorrelation per row
        """
        
        myRange = np.arange(min_sec,max_sec+bin_size_sec,bin_size_sec)
        self.st_autocorrelation = np.ndarray((len(self.neuron_list),myRange.shape[0]-1)) # to store the results
        
        for i,n in tqdm(enumerate(self.neuron_list)):
            n.spike_train.spike_time_autocorrelation(bin_size_sec = bin_size_sec, min_sec = min_sec, max_sec = max_sec)
            self.st_autocorrelation[i,:] = n.spike_train.st_autocorrelation_histogram[0]
        self.st_autocorrelation_bins = myRange

    
    def instantaneous_firing_rate(self, bin_size_sec = 0.02, sigma = 1, outside_interval_solution="remove"):
        """
        Calculates the instantaneous firing rate of the neurons in the Cell_group. 
        This is the firing rate of the neuron over time.
        The spikes are counted in equal sized time windows. (histogram)
        Then the spike count array is smooth with a gaussian kernel. (convolution)
        
        Arguments:
        bin_size_sec: Bin size in sec
        sigma: Standard deviation of the gaussian filter smoothing, values are in bins
        outside_interval_solution: What to do with time windows that are outside the set intervals. "remove" or "nan" are accepted.
        
        Returns:
        Saves self.ifr, self.ifr_rate and ifr_bin_size_sec 
        self.ifr is a tupple containing the ifr and time of the mid point of the bin.
        """
        
        ifrList = []
        for i, n in tqdm(enumerate(self.neuron_list)):
            n.spike_train.instantaneous_firing_rate(bin_size_sec=bin_size_sec,sigma=sigma,outside_interval_solution=outside_interval_solution)
            ifrList.append(n.spike_train.ifr[0])
            if(i == 0):
                time = n.spike_train.ifr[2]
                
        self.ifr = (np.stack(ifrList,axis=0),time)