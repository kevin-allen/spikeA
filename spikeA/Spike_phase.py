from spikeA.Spike_train import Spike_train
from spikeA.Intervals import Intervals
import spikeA.spike_time
import numpy as np
import matplotlib.pyplot as plt



class Spike_phase:
    """
    Class used to do analyse how oscillations modulate the firing of a neuron.
        
    It was developped to analyze the preferred theta phase of neurons but could be used for other oscillations as well. 
    
    You give it a spike_train and some cycles and then you can get 
    1. the phase of the spikes, from -np.pi to np.pi
    2. the mean vector length of the spikes
    3. the preferred phase of the neuron
    4. a histogram with the firing rate as a function of theta phase. 
    
    If you want to limit the analysis to certain time periods, you should use self.set_intervals so that cycles outside the intervals are not considered.
    
    Typical usage:
    
    mouse="bt8564"
    date="23072021"
    name=f"{mouse}-{date}-0105"
    path=f"/adata/electro/{mouse}/{name}/"
    ses = Kilosort_session(name=name, path = path)
    ses.load_parameters_from_files()
    theta = Theta(session=ses)
    theta.load_theta_epochs_and_cycles()
    stl = Spike_train_loader()
    stl.load_spike_train_kilosort(ses)
    cg = Cell_group(stl)
    st = cg.neuron_list[5].spike_train
    sph = Spike_phase(st,theta.cycles[0])
    sph.spike_phase()
    h = sph.spike_phase_histogram()
    pp, mvl = sph.spike_phase_stats()
    
    See docs_notebooks/spike_phase.ipynb for more information
    
    Attributes:
        spike_train: a spikeA.Spike_train. The time in the Spike_train.st array should be in seconds.
        cycles: a 2D numpy array containing beginning and end of theta cycles. The time in this array should be in seconds.
        phase: phase of spikes from -np.pi to np.pi
        intervals: intervals that can be used to only consider cycles within the intervals
        
    Methods:
        set_intervals()
        unset_intervals()
        spike_phase()
        spike_phase_histogram()
        spike_phase_stats()
        
    """
    def __init__(self,spike_train,cycles):
        """
        Constructor of the Spike_phase Class
        """
        
        if not isinstance(cycles, np.ndarray):
            raise TypeError("cycles argument of the Spike_phase constructor should be a numpy.ndarray but was {}".format(type(cycles)))
        
        
        self.spike_train = spike_train
        self.phase = None
        
        # to keep the original data when we apply intervals
        self.cycles_ori = cycles.copy(order='C') # so that we can pass it to c code
        
        self.cycles = self.cycles_ori
        
        # set default intervals that cover all cycles
        self.intervals = Intervals(inter=np.array([[0,self.cycles[-1,1].max()+1]]))
        
        return
          
    def set_intervals(self,inter):
        """
        Function to limit the analysis to cycle within a set of set specific time intervals
        
        Arguments:
        inter: 2D numpy array, one interval per row, time in seconds
        
        Return:
        The function will set self.intervals to the values of inter
        
        """
        
        self.intervals.set_inter(inter)
        self.cycles_inter = self.intervals.cycles_within_intervals(self.cycles_ori)
        # self.cycles is now pointing to self.cycles_inter
        self.cycles = self.cycles_inter
        
    
    def unset_intervals(self):
        """
        Function to remove the previously set intervals. 
        
        After calling this function, all spikes of the original spike train will be considered.
        The default interval that includes all spikes is set.
        """
        
        self.cycles = self.cycles_ori
        # set default time intervals from 0 to just after the last spike
        self.intervals = Intervals(inter=np.array([[0,self.cycles[-1,1].max()+1]]))
    
    
    def spike_phase(self):
        """
        Method to calculate the phase of spikes
        
        The c function set invalid to -1.0 and the valid phase from 0 to 2*np.pi
        The python code then set invalid to np.nan and the range from -np.pi to np.pi
        """
        
        # memory to store the results
        self.phase = np.empty_like(self.spike_train.st)
        
        spikeA.spike_time.spike_phase_func(self.spike_train.st,self.cycles,self.phase)
        
        self.phase[self.phase==-1.0] = np.nan
        self.phase = self.phase-np.pi
    
    def spike_phase_histogram(self,n_bins=36, rate = True):
        """
        Return a histogram with the firing rate as a function of phase
        
        Arguments:
        n_bins: Number of bins in the histogram
        rate: Whether to return the firing rate instead of the count
        """
        if self.phase is None:
            raise TypeError("self.phase is None, try calling Spike_phase.spike_phase before.")
        
        time_in_cycles = np.sum(self.cycles[:,1]-self.cycles[:,0])
        time_per_bin = time_in_cycles/n_bins
        
        bins = np.linspace(-np.pi,np.pi,n_bins)
        h = np.histogram(self.phase,bins=bins)
        
        if rate:
            H = h[0]/time_per_bin,h[1]
        else:
            H = h
        return H
        
  
    def spike_phase_stats(self):
        """
        Calculate circular mean and mean vector length of the spike phase
        ##Calculate Rayleigh p-value and Rayleigh z## (p value mostly 0 anyway...)
        """
        
        if self.phase is None:
            raise TypeError("self.phase is None, try calling Spike_phase.spike_phase before.")
        
        
        # get rid of np.nan
        phase = self.phase[~np.isnan(self.phase)]
        
        x = np.mean(np.cos(phase))
        y = np.mean(np.sin(phase))
        
        preferred_phase = np.arctan2(y,x)
        
        mean_vector_length = np.sqrt(x**2+y**2)
        
        ##Rayleigh stats
        #n = np.sum(np.ones_like(phase), axis=0)
#
        ## compute Rayleigh's R
        #R = n * mean_vector_length
#
        ## compute Rayleigh's z
        #z_rayleigh = R ** 2 / n
#
        ## compute p value using approxation in Zar, p. 617
        #p_rayleigh = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
#
#
        return preferred_phase,mean_vector_length#,p_rayleigh,z_rayleigh
        
        
    def plot_phase_histogram(self,ax):
        """
        Function to plot the phase histogram of a neuron
        
        """
        if self.phase is None:
            raise TypeError("self.phase is None, try calling Spike_phase.spike_phase before.")
        
        h = self.spike_phase_histogram()
        pp, mvl = self.spike_phase_stats()
        midBin = h[1][:-1]+ (h[1][1]-h[1][0])/2
        ax.plot(midBin,h[0])
        ax.set_ylim(0,np.max(h[0])+0.5)
        ax.set_xlabel("Theta phase (radian)")
        ax.set_ylabel("Firing rate (Hz)")
        if pp < 0:
            x = 0.1
        else:
            x = 0.5
            
        ax.text(x, 0.2, "mvl: {:.3f}".format(mvl), transform=ax.transAxes)
        ax.text(x, 0.3, "mean: {:.3f}".format(pp), transform=ax.transAxes)
        #ax.text(x, 0.1, "p rayleigh: {:.3f}".format(p), transform=ax.transAxes)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        