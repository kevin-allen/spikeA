"""
File containing the definition of the Spike_train class
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from spikeA.Intervals import Intervals
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import matplotlib.pyplot as plt 

class Spike_train:
    """
    Class representing the spikes recorded from one neuron.
    
    The class does the analysis of spike train. A spike train is defined as the spike in time emitted by a single neuron
    
    The spike train data can come from data files (e.g., outputs of Klustakwik) or generated data (e.g., generate_poisson_spike_train() function).
    To get data from files, have a look at the Spike_train_loader class.
    
    The time values of the spike trains are in seconds.
    
    Attributes:
    
    name: name for the spike train
    st: 1d numpy arrays containing the spike time a neuron. The time is in seconds.
    ifr: 1d numpy array containing the instantaneous firing rate of the neuron
    isi: 1d numpy array containing the inter-spike intervals between subsequent spikes
    
    sta: 1d numpy array containing the spike-time autocorrelation of the neuron
    m_rate: mean firing rate of the neuron

    Methods:
    set_spike_train(): set a spike train 
    generate_poisson_spike_train(): 
    generate_modulated_poisson_spike_train():
    n_spikes(): return the number of spikes of the neuron
    mean_firing_rate(): return the mean firing rate of the neuron
    inter_spike_intervals(): calculate the inter-spike intervals of the neuron
    inter_spike_intervals_histogram()
    inter_spike_interval_histogram_plot()
    instantaneous_firing_rate(): calculate the instantaneous firing rate in time
    instantaneous_firing_rate_autocorrelation():
    instantaneous_firing_rate_power_spectrum():
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
        
        # if a spike train was passed to the constructor, set the spike train
        if st is not None:
            self.set_spike_train(st=st)
        else:
            self.st = None # we can use this to know if st has been set or not (whether it is None or none)
            
    
    def set_spike_train(self,st):
        """
        Set the st and sampling_rate attribute of the Spike_train object
        
        Arguments
        st: 1D numpy array containing spike times in seconds 
        """
        # check that st is a numpy array of 1 dimension
        if not isinstance(st, np.ndarray):
            raise TypeError("st argument of the Spike_time constructor should be a numpy.ndarray but was {}".format(type(st)))
        if st.ndim != 1: # it should be one neuron so one dimention
            raise ValueError("st arguemnt of the Spike_time constructor should be a numpy array with 1 dimension but had {}".format(st.ndim))
            
        self.st = st
        
        print("Spike_train, name: {}, number of spikes {}, first: {}, last: {}".format(self.name, self.st.shape[0],self.st.min(),self.st.max()))
        
        # set default time intervals from 0 to just after the last spike
        self.intervals = Intervals(inter=np.array([[0,self.st.max()+1/self.sampling_rate]]),
                                           sampling_rate=self.sampling_rate)
        
        print("Total interval time: {} sec".format(self.intervals.total_interval_duration_seconds()))
        
        
    def generate_poisson_spike_train(self,rate_hz=20, sampling_rate=20000, length_sec=2):
        """
        Generate a spike train from a random poisson distribution
        
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
        Return the number of spikes of the cluster
        """
        if self.st is None:
            raise ValueError("set the spike train before using Spike_train.n_spike()")
        return self.st.shape[0]
    
    def mean_firing_rate(self):
        """
        Calculate the mean firing rate (number of spikes / sec) of each cluster
        Use the total time in seconds from self.intervals

        Return the mean firing rate
        """
        if self.st is None:
            raise ValueError("set the spike train before using Spike_train.mean_firing_rate()")
        return self.n_spikes() / self.intervals.total_interval_duration_seconds()
        
    def inter_spike_intervals(self):
        """
        Calculate the inter spike intervals

        The results are stored in a 1D numpy called self.isi
        self.isi should have a length of len(self.st) -1
        
        Return
        The results is stored in self.isi
        """
        if self.st is None:
            raise ValueError("set the spike train before using Spike_train.inter_spike_intervals()")
        self.isi = np.diff(self.st)
        
    def inter_spike_intervals_histogram(self, bin_size_ms=5,max_time_ms=2000, density= False):
        """
        Calculate an inter spike interval histogram
        Save in self.isi_histogram
        
        Return
        The results are stored in self.isi_histogram, which is the output of np.histogram
        """
        self.inter_spike_intervals()
        isi_ms = self.isi*1000
        self.isi_histogram = np.histogram(isi_ms, bins= np.arange(0,max_time_ms+bin_size_ms,bin_size_ms),density= density)
        self.isi_histogram_density = density
    
    def inter_spike_interval_histogram_plot(self, type = "line"):
        """
        Plot the inter spike interval histogram using matplotlib

        Call this method by self.inter_spike_interval_histogram_plot()
        """
        if self.isi_histogram is None:
            raise ValueError("please run inter_spike_intervals_histogram() first")  
        
        x = self.isi_histogram[1]
        diff = x[1] - x[0]
        median = diff/2
        
        timestamp = x[:-1] + median    
    
        if type == "bar":
            plt.bar(timestamp, self.isi_histogram[0], width = diff) 
        else:
            plt.plot(timestamp, self.isi_histogram[0])

        plt.xlabel("ms")
        
        if self.isi_histogram_density is True:
            plt.ylabel("density")
        else:
            plt.ylabel("spikes")
            
        #pass

    
    def instantaneous_firing_rate(self,bin_size_ms = 1, sigma = 1):
        """
        Calculate the instantaneous firing rate. This is the firing rate of the neuron over time.

        The spikes are counted in equal sized time windows. (histogram)
        Then the spike count array is smooth with a gaussian kernel. (convolution)
        The result is a tuple containing the ifr, count and edges. These are 1D numpy arrays
        """    
        st_ms=self.st*1000
        count, edges = np.histogram(st_ms, bins = np.arange(0, self.intervals.total_interval_duration_seconds() * 1000+bin_size_ms, bin_size_ms))
        
        # from spike count to rate 
        hz = count / (bin_size_ms / 1000)
        
        ifr = gaussian_filter1d(hz.astype(np.float32), sigma = sigma)
        
        self.ifr = ifr,count,edges
        self.ifr_rate = 1000/bin_size_ms
                
      
    def instantaneous_firing_rate_autocorrelation(self, normed= False, maxlags= 200):
        """
        Calculate the autocorrelation of the instantaneous firing rate array (self.isi)
        
        Save the results in self.ifr_autocorrelation
        """
        res= np.correlate(self.ifr[0],self.ifr[0],mode='full')
        if normed== False:
            self.ifr_autocorrelation= res[int(res.size/2-maxlags):int(res.size/2+maxlags)]
        else: 
            self.ifr_autocorrelation= self.ifr_autocorrelation/np.max(self.ifr_autocorrelation)
        
    def instantaneous_firing_rate_power_spectrum(self, nfft = None, scaling = "density"):
        """
        Calculate the power spectrum of the instantaneous firing rate array (self.ifr)
        
        Save the results in self.ifr_power_spectrum
        """
        if self.ifr is None:
            raise ValueError("Please run the instantaneous_firing_rate() first")
        
        f, psd = signal.periodogram(self.ifr[0],fs=self.ifr_rate, nfft = nfft)
        self.ifr_power_spectrum = f, psd
  
        
    def instantaneous_firing_rate_power_spectrum_plot(self):
        """
        Plot the power spectrum of the instantaneous firing rate (self.ifr_power_spectrum)
        
        """
        if self.ifr_power_spectrum is None:
            print("Need to run instantaneous_firing_rate_power_spectrum first")
            
        plt.plot(self.ifr_power_spectrum[0], self.ifr_power_spectrum[1])
        plt.xlabel("Hz")
        plt.ylabel("PSD (s**2/Hz)")
        
        