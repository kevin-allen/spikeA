"""
File containing the definition of the Spike_train class
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from spikeA.Intervals import Intervals
import spikeA.spike_time # c extension
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from numba import njit, prange
import matplotlib.pyplot as plt 

class Spike_train:
    """
    Class representing the spikes recorded from one neuron.
    
    The class does the analysis of spike train. A spike train is defined as the spike in time emitted by a single neuron
    
    The time values of the spike trains are in seconds.
    
    You can set some time intervals to limit the analysis to some portion of the recordings
        
    Attributes:
    
    name: name for the spike train
    st: 1d numpy arrays containing the spike time of a neuron. The time is in seconds. st is a pointer to st_ori or st_inter
    ifr: 1d numpy array containing the instantaneous firing rate of the neuron
    isi: 1d numpy array containing the inter-spike intervals between subsequent spikes
    
    # don't use the next two when programming, use st
    st_ori: 1d numpy array containing the spike time of a neuron, not affected by setting time intervals
    st_inter: 1d numpy array containing the spike time of a neuron within the set time intervals
    
    sta: 1d numpy array containing the spike-time autocorrelation of the neuron
    m_rate: mean firing rate of the neuron
    ...

    Methods:
    n_spikes(): return the number of spikes of the neuron
    mean_firing_rate(): return the mean firing rate of the neuron
    inter_spike_intervals(): calculate the inter-spike intervals of the neuron
    instantaneous_firing_rate(): calculate the instantaneous firing rate in time
    inter_spike_intervals_histogram(): Calculate an inter spike interval histogram
    inter_spike_interval_histogram_plot(): Plot the inter spike interval histogram using matplotlib
    mid_point_from_edges(): Find the middle point of the edges of bins(output of np.histogram) and reduce the number of edges by 1.
    instantaneous_firing_rate(): Calculate the instantaneous firing rate.
    instantaneous_firing_rate_autocorrelation(): Calculate the autocorrelation of the instantaneous firing rate array.
    instantaneous_firing_rate_autocorrelation_plot(): Plots the instantaneous firing rate autocorrelation using matplotlib.
    instantaneous_firing_rate_power_spectrum(): Calculate the power spectrum of the instantaneous firing rate array.
    instantaneous_firing_rate_crosscorelation(): Calculate the crosscorrelation of the instantaneous firing rate array.
    instantaneous_firing_rate_crosscorelation_plot(): Plots the crosscorrelation of the instantaneous firing rate.
    instantaneous_firing_rate_power_spectrum_plot(): Plots the power spectrum of the instantaneous firing rate using matplotlib.
    spike_time_autocorrelation(): This function calculates the spike-time autocorrelation.
    spike_autocorrelation_plot(): Plots the spike-time autocorrelation using matplotlib.
    spike_time_crosscorrelation(): This function calculate the spike-time crosscorrelation between 2 spike trains.
    refractory_period_ratio(): Score that indicates whether the spike train has a clean refractory period.
    """
    def __init__(self,name=None, sampling_rate = 20000, st = None):
        """
        Constructor of the Spike_train class

        Arguments:
        name: Name for the Spike_train object
        sampling_rate: sampling rate of recording. This is used when you want the sample value of spikes
        st: 1d numpy array with the spike time of one neuron. The values you pass in should be in seconds.
        """      
        
        # assign argument of function to the object attributes
        self.name = name
        self.sampling_rate = sampling_rate
        self.intervals = None
        
        # if a spike train was passed to the constructor, set the spike train
        if st is not None:
            self.set_spike_train(st=st)
        else:
            self.st = None # we can use this to know if st has been set or not (whether it is None or none)
        
    
    def set_spike_train(self,st, verbose= False):
        """
        Set the st attribute of the Spike_train object
        
        This will reset the time intervals to the default intervals including all spikes
        
        Arguments
        st: 1D numpy array containing spike times in seconds 
        """
        # check that st is a numpy array of 1 dimension
        if not isinstance(st, np.ndarray):
            raise TypeError("st argument of the Spike_time constructor should be a numpy.ndarray but was {}".format(type(st)))
        if st.ndim != 1: # it should be one neuron so one dimention
            raise ValueError("st arguemnt of the Spike_time constructor should be a numpy array with 1 dimension but had {}".format(st.ndim))
            
        # the original spike train
        self.st_ori = st
        
        # self.st is a pointer to self.st_ori
        self.st = self.st_ori
        
        
        # set default time intervals from 0 to just after the last spike
        if self.intervals is not None:
            self.set_intervals(inter=np.array([[0,self.st.max()+1]]))
        else :
             # get intervals for the first time
            self.intervals = Intervals(inter=np.array([[0,self.st.max()+1]]))
        if verbose:
            print("Spike_train, name: {name}, number of spikes: {n_spikes:{fill}{align}{width}} mean firing rate: {rate:.2f} Hz".format(name=self.name, n_spikes=self.n_spikes(),fill=" ",align="<", width=8,rate=self.mean_firing_rate()))
        
    def set_intervals(self,inter):
        """
        Function to limit the analysis to spikes within a set of set specific time intervals
        
        Note on memory use:
        When intervals are not set, self.st points to self.st_ori (numpy array with all spikes)
        When you use set_intervals(), a new numpy array is stored in self.st which contains only the spikes within the intervals. self.st_ori is kept as a backup of the complete spike train.
        A side effect of calling set_intervals() is that the memory size of the Spike_train object will increase. You can return the size to the original size by calling unset_intervals()
        
        Arguments:
        inter: 2D numpy array, one interval per row, time in seconds
        
        Return:
        The function will set self.intervals to the values of inter
        
        """
        
        if self.st is None:
            raise ValueError("the spike train should be set before setting the intervals")
        
        self.intervals.set_inter(inter)
        self.st_inter = self.intervals.spike_train_within_intervals(self.st_ori)
        # self.st is now pointing to self.st_inter
        self.st = self.st_inter
        #print("Number of spikes: {}".format(self.st.shape[0]))
    
    def unset_intervals(self):
        """
        Function to remove the previously set intervals. 
        
        After calling this function, all spikes of the original spike train will be considered.
        The default interval that includes all spikes is set.
        """
        if self.st is None:
            raise ValueError("the spike train should be set before setting the intervals")
        
        self.st = self.st_ori.copy() # create a copy of our st_ori, not a pointer
        # set default time intervals from 0 to just after the last spike
        self.intervals.set_inter(inter=np.array([[0,self.st.max()+1]]))
        #print("Number of spikes: {}".format(self.st.shape[0]))
        
    def generate_poisson_spike_train(self,rate_hz=20, sampling_rate=20000, length_sec=2):
        """
        Generate a spike train from a random poisson distribution.
        
        Arguments
        rate_hz: Firing rate of the spike train
        sampling_Rate: sampling rate for the poisson process
        length_sec: length of the spike train (sampling process in the poisson distribution)
        
        Results are stored in self.st
        """
        # check that sampling_rate value makes sense
        if sampling_rate <= 0 or sampling_rate > 100000:
            raise ValueError("sampling_rate arguemnt of the Spike_time constructor should be larger than 0 and smaller than 100000 but was {}".format(sampling_rate))
        # check that sampling_rate value makes sense
        if rate_hz < 0:
            raise ValueError("rate_hz argument of Spike_train.generate_poisson_spike_train() should not be negative but was {}".format(rate_hz))
        if length_sec < 0:
            raise ValueError("length_sec argument of Spike_train.generate_poisson_spike_train() should not be negative but was {}".format(length_sec))
        
        # variables to sample the poisson distribution
        length = sampling_rate*length_sec
        mu = rate_hz/sampling_rate # rate for each sample from the poisson distribution
        st = np.nonzero(poisson.rvs(mu, size=length))[0] # np.nonzero returns a tuple of arrays, we get the first element of the tuple
       
        st = st/sampling_rate # to get the time in seconds
       
        self.set_spike_train(st = st)
    
    def generate_poisson_spike_train_from_rate_vector(self,mu, sampling_rate=20000):
        """
        Generate a spike train from a random poisson distribution.
        
        Arguments
        mu: Firing rate vector in Hz
        sampling_Rate: sampling rate for the poisson process
                
        Results are stored in self.st
        """
        # check that sampling_rate value makes sense
        if sampling_rate <= 0 or sampling_rate > 100000:
            raise ValueError("sampling_rate arguemnt of the Spike_time constructor should be larger than 0 and smaller than 100000 but was {}".format(sampling_rate))
        
        # variables to sample the poisson distribution
        mu = mu/sampling_rate # rate for each sample from the poisson distribution
        mu[mu<0.0] = 0.0
        st = np.nonzero(poisson.rvs(mu))[0] # np.nonzero returns a tuple of arrays, we get the first element of the tuple
        st = st/sampling_rate # to get the time in seconds
       
        self.set_spike_train(st = st)
        
    def generate_modulated_poisson_spike_train(self,rate_hz=50, sampling_rate=20000, length_sec=2,modulation_hz = 10, modulation_depth = 1,min_rate_bins_per_cycle=10,phase_shift=0):
        """
        Generate a spike train from a random poisson distribution in which the firing rate to follow a sine wave
        
        This can be used to test the method calculating a power spectrum from the instantaenous firing rate.
        
        Arguments
        rate_hz: Firing rate of the spike train
        sampling_Rate: sampling rate for the poisson process
        length_sec: length of the spike train (sampling process in the poisson distribution)
        modulation_hz: frequency of the modulation
        modulation_depth: depth of the firing rate modulation by the sine wave, 1 = will go from rate_hz*0 to rate_hz*2, 0.5 = will go from rate_hz*0.5 to rate_hz*1.5
        bins_per_cycle: how many times we are changing the firing rate frequency per cycle.
        phase_shift: to shift the phase of the modulation, 0 = no shift, np.pi/2 = 90 degree shift, np.pi = 180 degree shift.
        
        Results are stored in self.st
        """   
        
        # check that sampling_rate value makes sense
        if sampling_rate <= 0 or sampling_rate > 100000:
            raise ValueError("sampling_rate arguemnt of the Spike_time constructor should be larger than 0 and smaller than 100000 but was {}".format(sampling_rate))
        # check that sampling_rate value makes sense
        if rate_hz < 0:
            raise ValueError("rate_hz argument of Spike_train.generate_poisson_spike_train() should not be negative but was {}".format(rate_hz))
        if length_sec < 0:
            raise ValueError("length_sec argument of Spike_train.generate_poisson_spike_train() should not be negative but was {}".format(length_sec))
        if sampling_rate < modulation_hz*min_rate_bins_per_cycle :
            raise ValueError("sampling_rate is too low for the modulation frequency")
        if modulation_depth > 1:
            print("modulation_depth in generate_modulated_poisson_spike_train() should not be larger than 1, value set to 1")
            modulation_depth=1
        if modulation_depth < 0:
            print("modulation_depth in generate_modulated_poisson_spike_train() should not be negative, value set to 0")
            modulation_depth=0

        # how many samples per cycles
        samples_per_cycle = sampling_rate/modulation_hz
        # how many cycles in the spike train
        n_cycles = length_sec*modulation_hz
        
        # calculate the rate for all the samples within a cycle
        rates = (np.sin(np.arange(0,2*np.pi,2*np.pi/samples_per_cycle)+phase_shift) *modulation_depth+1)*rate_hz

        # sample from poisson distribution using our array of rates, stack the list of array into a matrix, then flatten matrix to 1D array
        res = np.stack([ poisson.rvs(r/sampling_rate,size=n_cycles) for r in rates]).flatten('F')

        # go from an array of 0 and 1 to an array with the spike times
        # note that we are throwing away spikes if there are more than one per sample!
        st = np.nonzero(res)[0]
        st= st/sampling_rate
        self.set_spike_train(st = st)
        
    def n_spikes(self):
        """
        This function calculates the total number of spikes.
        
        Arguments:
        st needs to be set before this function can be called.
       
        Return:
        the total number of spikes of the cluster.
        """
        if self.st is None:
            raise ValueError("set the spike train before using Spike_train.n_spikes()")
        return self.st.shape[0]
    
    def mean_firing_rate(self):
        """
        Calculate the mean firing rate (number of spikes / sec) of each cluster.
        Spike train needs to be set before this function can be called.
        Use the total time in seconds from self.intervals
        
        Arguments:
        st needs to be set before this function can be called.
        
        Return:
        The mean firing rate (spikes/seconds) is stored in self.n_spikes
        """
        if self.st is None:
            raise ValueError("set the spike train before using Spike_train.mean_firing_rate()")
        return self.n_spikes() / self.intervals.total_interval_duration_seconds()
        
    def inter_spike_intervals(self):
        """
        Calculate the inter spike intervals.
        
        Arguments:
        st needs to be set before this function can be called.
        
        Return:
        The inter spike intervals are calculated and stored as a 1D numpy array in self.isi
        self.isi should have a length of len(self.st) -1
        """
        if self.st is None:
            raise ValueError("set the spike train before using Spike_train.inter_spike_intervals()")
        self.isi = np.diff(self.st)
        
    def inter_spike_intervals_histogram(self, bin_size_ms=5,max_time_ms=500, density= False):
        """
        Calculate an inter spike interval histogram
        The inter_spike_intervals() method needs to be run before this method. 
        
        Arguments:
        Expects an 1D np.array from self.isi.
        bin_size_ms:
        max_time_ms:
        density:
        
        Return:
        The results are stored in self.isi_histogram, which is the output of np.histogram
        """
        self.inter_spike_intervals()
        isi_ms = self.isi*1000
        self.isi_histogram = np.histogram(isi_ms, bins= np.arange(0,max_time_ms+bin_size_ms,bin_size_ms),density= density)
        self.isi_histogram_density = density
    
    def inter_spike_interval_histogram_plot(self, plot_type = "line"):
        """
        Plot the inter spike interval histogram using matplotlib
        
        Arguments:
        Expects 2 np.arrays from self.isi_histogram.
        plot_type: can be set to line/bar to givout different plots.

        Returns:
        A line/bar plot of the inter spike intervals.
        """
        if self.isi_histogram is None:
            raise ValueError("please run inter_spike_intervals_histogram() before inter_spike_interval_histogram_plot() ")  
       
        timestamp = self.mid_point_from_edges(self.isi_histogram[1])
        diff = timestamp[1] - timestamp[0]
    
        if plot_type == "bar":
            plt.bar(timestamp, self.isi_histogram[0], width = diff) 
        else:
            plt.plot(timestamp, self.isi_histogram[0])

        plt.xlabel("Time (ms)")
        
        if self.isi_histogram_density is True:
            plt.ylabel("Density")
        else:
            plt.ylabel("Count")
            
        #pass
        
    def mid_point_from_edges(self, edges):
        """
        Find the middle point of the edges of bins (output of np.histogram) and therefore reduce the number of edges by 1.
        
        Arguments:
        edges: np.array containing the edges from np.histogram()
        
        Returns:
        A np.array containing the midpoint of every bin stored in the variable "timestamp".
        """
        x = edges
        diff = x[1] - x[0]
        median = diff/2
        
        timestamp = x[:-1] + median
        
        return timestamp
    

    def instantaneous_firing_rate(self,bin_size_sec = 0.001, sigma = 1, 
                                  shift_start_time=0, time_start=None, time_end=None, outside_interval_solution="remove"):
        """
        Calculates the instantaneous firing rate. This is the firing rate of the neuron over time.
        The spikes are counted in equal sized time windows. (histogram)
        Then the spike count array is smooth with a gaussian kernel. (convolution)
        
        Arguments:
        bin_size_sec: Bin size in sec
        sigma: Standard deviation of the gaussian filter smoothing, values are in bins

        shift_start_time: amount to add to the starting time for the calculation of IFR, for example if the IFR was to be calculated from 0 to 1000, then it will be calculated from 0+shift_start_time to 1000.
        time_start, time_end: manually define start/end to calculate IFR. useful to apply globally to several neurons so that a numpy array of regular shape holds the population IFR (use: outside_interval_solution='nan')
        outside_interval_solution: What to do with time windows that are outside the set intervals. "remove" or "nan" are accepted.
        
        Returns:
        Saves self.ifr, self.ifr_rate and ifr_bin_size_sec 
        self.ifr is a tupple containing the ifr, the count, and mid point of the bin.
        """    
        
        # call like from pose
        # n.spike_train.instantaneous_firing_rate(bin_size_sec = bin_size_sec, sigma= sigma_ifr, time_start=min(time)-bin_size_sec/2, time_end=max(time), outside_interval_solution="remove")
        # and check pose's time and IFR'mid time match: pose_time = ap.pose[:,0], ifr_mid_time = n.spike_train.ifr[2]

        if not(time_start is None or time_end is None):
            bins = np.arange(time_start, time_end+bin_size_sec, bin_size_sec)
        else:
            bins = np.arange(np.min(self.intervals.inter)+shift_start_time, np.max(self.intervals.inter)+bin_size_sec, bin_size_sec)
        
        #plt.hist(np.append(np.diff(bins),bin_size_sec+0.1),bins=50)
        #plt.title("bins in spike_train")
        #plt.show()
        
        count, edges = np.histogram(self.st, bins = bins)
        
        # from spike count to rate 
        hz = count / (bin_size_sec)
        
        ifr = gaussian_filter1d(hz.astype(np.float32), sigma = sigma)
        
        # we need to remove the time bins that are not in the intervals
        mid = self.mid_point_from_edges(edges)

        keep=self.intervals.is_within_intervals(mid, include_ties_nostrict=True)
               

        if outside_interval_solution == "remove":    
            self.ifr = ifr[keep],count[keep],mid[keep]    
        elif outside_interval_solution == "nan":
            ifr[~keep]=np.nan
            self.ifr = ifr,count,mid
            
        else:
            print("invalid value for argument outside_interval_solution")
            raise ValueError("set outside_interval_solution to remove or nan")

        self.ifr_rate = 1/bin_size_sec
        self.ifr_bin_size_sec= bin_size_sec
                
      
    def instantaneous_firing_rate_autocorrelation(self, normed= False, max_lag_sec= 1):
        """
        Calculates the autocorrelation of the instantaneous firing rate array (self.isi)
        
        Arguments:
        The instantaneous_firing_rate() arrays saved in self.ifr
        normed:
        max_lag_sec:
        
        Returns:
        The results are saved in self.ifr_autocorrelation
        """
        res= np.correlate(self.ifr[0],self.ifr[0],mode='full')
        maxlag= max_lag_sec/self.ifr_bin_size_sec
        res= res[int(res.size/2-maxlag):int(res.size/2+maxlag)]
        if normed== False:
            self.ifr_autocorrelation= res
        else: 
            self.ifr_autocorrelation= res/np.max(res)
        

    def instantaneous_firing_rate_autocorrelation_plot(self,timewindow=None):
        if timewindow== None:
            plt.plot(self.ifr_autocorrelation)
        else:
            plt.plot(np.arange(-timewindow,timewindow,1),self.ifr_autocorrelation)
                
                
    def instantaneous_firing_rate_power_spectrum(self, nperseg = 2**9, scaling = "spectrum"):
        """
        Calculates the power spectrum of the instantaneous firing rate
        
        Arguments:
        nperseg: The data is divided in overlapping segments. This is the segment length.
        scaling: can be 'spectrum' or 'density'
        
        Returns:
        Save the results in self.ifr_power_spectrum
        """

        f, ps = signal.welch(self.ifr[0],fs=self.ifr_rate, nperseg=nperseg, scaling=scaling)
        self.ifr_power_spectrum = f, ps
    
    
    def instantaneous_firing_rate_power_spectrum_plot(self):
        """
        Plot the power spectrum of the instantaneous firing rate (self.ifr_power_spectrum)
        
        Arguments:
        2 numpy arrays from self.ifr_power_spectrum
        
        Returns:
        A plot of the ifr power spectrum.
        """
        if self.ifr_power_spectrum is None:
            print("Need to run instantaneous_firing_rate_power_spectrum first")
            
        plt.plot(self.ifr_power_spectrum[0], self.ifr_power_spectrum[1])
        plt.xlabel("Hz")
        plt.ylabel("PSD (s**2)") #assuming that scaling='spectrum'; otherwise use s**2/Hz
        
        
    def instantaneous_firing_rate_crosscorrelation(self,spike2=None,normed= False, max_lag_sec= 0.2):
        """
        Calculates the instantaneous firing rate crosscorrelation.
        
        Arguments:
        spike2: second Spike train object to correlate with
        normed: normalize to max correlation
        max_lag_sec: max delta time in seconds between two cells
        The instantaneous_firing_rate() arrays saved in self.ifr
        
        Returns:
        The results are saved in self.ifr_corsscorrelation (normed or not normed).
        """
        if spike2 is None:
            spike2 = Spike_train(name= "spike2", sampling_rate= 20000,st=np.arange(0,10000))
        
        spike2.set_spike_train(spike2.st)
        spike2.inter_spike_intervals()
        spike2.instantaneous_firing_rate()
        
        res= np.correlate(self.ifr[0],spike2.ifr[0],mode='full')
        maxlag= max_lag_sec/self.ifr_bin_size_sec
        res= res[int(res.size/2-maxlag):int(res.size/2+maxlag)]
        if not normed:
            self.ifr_crosscorrelation = res
        else:
            self.ifr_crosscorrelation = res/np.max(res)
            
    def instantaneous_firing_rate_crosscorrelation_plot(self,timewindow=None):
        """
        Plots the instantaneous firing rate crosscorrelation.
        
        Arguments:
        ifr crosscorrelation: saved in self.ifr_corsscorrelation (normed) or self.ifr_crosscorrelation (not normed)
        timewindow: can be defined to evaluate certain time windows of the spike train? Set to None by default.
        
        Returns:
        A plot of the ifr crosscorrelation.
        """
        if timewindow== None:
            plt.plot(self.ifr_crosscorrelation)
        else:
            plt.plot(np.arange(-timewindow,timewindow,1),self.ifr_crosscorrelation)
        

        if self.ifr is None:
            raise ValueError("Please run the instantaneous_firing_rate() first")
        
        f, psd = signal.periodogram(self.ifr[0],fs=self.ifr_rate, nfft = nfft)
        self.ifr_power_spectrum = f, psd
  
        
    
        
    def spike_time_autocorrelation(self,bin_size_sec=0.0005, min_sec=-0.1, max_sec=0.1):
        """
        This function calculate the spike-time autocorrelation by comparing the inter-spike intervals between all possible pair of spikes that fall in the 0-range_ms. 
        Each spike is treated in turn as the reference spike.
        The intervals between the reference spike and the subsequent spikes are calculated and binned in an histogram.
        It only calculates time intervals between the reference spike and spikes that occurred later in time.
        
        I moved the calculation to c to make it faster 
        
        Arguments
        bin_size_sec: bin size of the histogram
        min_sec: How far back from the reference spike are we considering
        max_sec: How far after from the reference spike are we considering
        
        Return
        The histogram is stored in self.st_autocorrelation_histogram as a tuple, (count, edges)
        """        
       
        if not isinstance(self.st,np.ndarray):
            raise TypeError("self.st should be a np.ndarray")
            
        myRange = np.arange(min_sec,max_sec+bin_size_sec,bin_size_sec)
        myHist = np.zeros(myRange.shape[0]-1) # to store the results
        
        spikeA.spike_time.spike_time_autocorrelation_func(self.st,myHist,min_sec,max_sec,bin_size_sec)
        self.st_autocorrelation_histogram = myHist,myRange

    def spike_autocorrelation_plot(self, plot_type = "line"):

        """
        Plot the spike_autocorrelation using matplotlib.
        
        Arguments:
        Expects 2 numpy arrays from self.st_autocorrelation_histogram
        plot_type: can be set to a bar or a line diagram
        
        Returns:
        A bar/line plot of the spike autocorrelation.
        """
    
        if self.st_autocorrelation_histogram is None:
            raise ValueError("please run inter_spike_intervals_histogram() before inter_spike_interval_histogram_plot() ")

        timestamp = self.mid_point_from_edges(self.st_autocorrelation_histogram[1])
    
        if plot_type == "bar":
            plt.bar(timestamp, self.st_autocorrelation_histogram[0]) 
        else:
            plt.plot(timestamp, self.st_autocorrelation_histogram[0])
        plt.ylim(0,np.max(self.st_autocorrelation_histogram[0]))
        plt.xlabel("Time (ms)")
        plt.ylabel("Count")
        
        
    def spike_time_crosscorrelation(self,st1=None,st2=None, bin_size_sec=0.0005, min_sec=-0.1, max_sec=0.1):
        """
        This function calculates the spike-time crosscorrelation between 2 Spike_train objects.
        
        Arguments:
        st1: np.array with spike times, if not provided, the spike times of the object will be used. These are the reference spikes
        st2: np.array with spike times
        bin_size_sec: size of bins in the histogram
        min_sec: minimum value in the histogram
        max_sec: maximum value in the histogram
        
        Returns:
        Tuple containing the histogram. The first element is the count and seconds are the edges of the histogram bin
        """
        if st1 is None:
            st1=self.st
        
        if not isinstance(st1,np.ndarray):
            raise TypeError("st1 should be a np.ndarray")
        
        if st2 is None:
            raise ValueError("you need to provide a second series of spike times")
        
        st2New = st2
        if isinstance(st2New,Spike_train):
            st2New=st2New.st # use the spike times
        
    #    if not isinstance(st2New,np.ndarray):
    #        raise TypeError("st2New should be a np.ndarray")
        
        myRange = np.arange(min_sec,max_sec+bin_size_sec,bin_size_sec)
        myHist = np.zeros(myRange.shape[0]-1) # to store the results

        
        spikeA.spike_time.spike_time_crosscorrelation_func(st1,st2New,myHist,min_sec,max_sec,bin_size_sec)
        
        return (myHist,myRange)
    
    
    def refractory_period_ratio(self, bin_size_sec=0.0005, min_sec=0.0, max_sec=0.03,refractory_length_sec=0.002,outside_refractory_min_sec=0.004):
        """
        Calculate a ratio between the number of spikes within the refractory period and those outside. 
        A spike-time autocorrelation is calculate and the mean number of spikes within the refractory period is devided by the mean number of spikes outside of it.
        
        The function calls spike_time_autocorrelation to construct the spike-time autocorrelation.
        
        Arguments:
        bin_size_sec: bin size in the spike-time autocorrelation
        min_sec: minimum time in the spike-time autocorrelation
        max_sec: maximum time in the spike-time autocorrelation
        refractory_length_sec: length of the refractory period in seconds
        outside_refractory_min_sec: time from which we are clearly outside the refractory period
        
        Return
        Refractory period ratio
        
        """
        self.spike_time_autocorrelation(bin_size_sec=bin_size_sec, min_sec=min_sec, max_sec=max_sec)
        timestamp =self.st_autocorrelation_histogram[1][1:] # larger edge of each bin
        meanRefractory = self.st_autocorrelation_histogram[0][np.logical_and(timestamp>0,timestamp<=refractory_length_sec)].mean()
        meanOutside = self.st_autocorrelation_histogram[0][timestamp>outside_refractory_min_sec].mean()
        
        if meanOutside ==0 :
            return np.nan
        else:
            return meanRefractory/meanOutside
        
            
