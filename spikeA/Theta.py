from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import signal, ndimage
import scipy.stats as stats
from tqdm import tqdm
import pickle
from os import path


from spikeA.Dat_file_reader import Dat_file_reader
from spikeA.Session import Kilosort_session

class Theta:
    """
    Class used to analyze theta oscillations. Mainly for the detection of theta epochs and theta cycles
    
    
    Unless you want to look under the hood, you probably want to run detect_theta_cycles_one_session() to get the theta epochs and theta cycles for one recording session.
    
    Common usage of this class
    
    # create a theta object
    mouse="bt8564"
    date="23072021"
    name=f"{mouse}-{date}-0105"
    path=f"/adata/electro/{mouse}/{name}/"
    ses = Kilosort_session(name=name, path = path)
    ses.load_parameters_from_files()
    theta = Theta(session=ses)
    
    # detect theta epochs and cycles
    # this will take 15-30 minutes because it has to read the .dat files
    theta.detect_theta_cycles_one_session()
    
    # load previously detected epochs and cycles
    theta.load_theta_epochs_and_cycles()
    
    # to get access to epochs and cycles
    theta.epochs, theta.cycles
    
    
    See also spikeA/docs_notebooks/LFP_theta_cycles.ipynb
    
    Attributes:
        session: a spikeA.Session object
        epochs: a dictionary with one key per channel analyzed. For each channel, it contains a 2D numpy array with the beginning and end of an epoch on each row. The time is in seconds.
        cycles: a dictionary with one key per channel analyzed. For each channel, it contains a 2D numpy array with the beginning and end of a cycle on each row. The time is in seconds.
        
    Methods:
        butter_bandpass()
        butter_bandpass_filter()
        filter_frequency_response_plot()
        calculate_oscillation_power()
        detectThetaEpochs()
        detect_cycles()
        detect_cycles_in_epochs()
        detect_theta_cycles_one_channel()
        detect_theta_cycles_one_session()
    """
    def __init__(self,session):
        """
        Constructor of the Theta Class
        """
        
        self.session = session
        self.epochs = None
        self.cycles = None
        
        self.min_theta_frequency=4
        self.max_theta_frequency=12
        self.sampling_rate=20000
        self.order=2
        self.conv_kernel_sigma_ms=200 # for filtering and theta power calculation
        self.min_delta_frequency=2
        self.max_delta_frequency=4 # for filtering and delta power calculation 
        self.theta_delta_ratio_threshold=2
        self.theta_epoch_min_length_ms = 500
        
        
        return
      
    def butter_bandpass(self,lowcut, highcut, fs, order=2):
        """
        Create a Butterworth bandpass filter

        Arguments:
        lowcut: minimal frequency for filter in Hz
        highcut: maximal frequency for filter in Hz
        fs: sampling rate of the signal in Hz
        order: order of the filter

        Returns:
        the filter
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        sos = signal.butter(order, [low, high], btype='band' ,  output='sos')
        return sos
    
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=2):
        """
        Filter data with a Butterworth bandpass filter

        Arguments:
        data: 1D numpy array containing the signal
        lowcut: minimal frequency for filter in Hz
        highcut: maximal frequency for filter in Hz
        fs: sampling rate of the signal in Hz
        order: order of the filter
        """
        sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.sosfilt(sos, data)
        return y 
    

    def filter_frequency_response_plot(self,myFilter,fs, xlim=(0,20)):
        """
        Plot the frequency response of a filter
        """

        w, h = signal.sosfreqz(myFilter, worN=1500)
        db = 20*np.log10(np.maximum(np.abs(h), 1e-5)) # don't worry about this, transforms h to dB unit
        plt.plot(w/np.pi*fs/2, db) # first part transform w to in Hz unit
        plt.ylim(-200, 5)
        plt.xlim(xlim[0],xlim[1])
        plt.ylabel('Gain (dB)')
        plt.xlabel('Frequency (Hz)')
        plt.title('Frequency Response')
        plt.show()


    def calculate_oscillation_power(self,data, min_frequency=4, max_frequency=12, sampling_rate=20000,order=2,conv_kernel_sigma_ms=200, fft = True):
        """
        Function to calculate the power in time at a specific frequency.

        Steps:
        1. Band-pass filter the signal
        2. Get abs value of signal
        3. Convolution of abs signal with gaussian kernel
        4. Return the results

        Arguments:
        data: 1D numpy array with the signal you want to process
        min_frequency: minimal frequency of the bandpass filter in Hz
        max_frequency: maximal frequency of the bandpass filter in Hz
        sampling_rate: sampling rate of the signal in Hz
        order: order of the filter
        conv_kernel_sigma_ms: standard deviation of the gaussian kernel to convolute the np.abs(filteredData)
        fft: whether to use fft to do the convolution

        Return:
        A 1D numpy array of the size of data containing the power at a given frequency range in time
        """

        # filter the signal
        myFilter = self.butter_bandpass(min_frequency, max_frequency,fs = sampling_rate,order=order)
        y = self.butter_bandpass_filter(data,min_frequency, max_frequency,fs = sampling_rate,order=order)

        # get power in time
        yrs= np.sqrt(y**2)
        convKernelSigmaSamples= int((sampling_rate/1000)*conv_kernel_sigma_ms)


        if fft :
            std=5
            x = np.linspace(-convKernelSigmaSamples*std,convKernelSigmaSamples*std,convKernelSigmaSamples*std*2)
            gKernel = stats.norm.pdf(x=x,loc=0, scale=convKernelSigmaSamples)
            conv = signal.fftconvolve(yrs,gKernel,mode="same")


        else:  # this convolution analysis is really slow for large data set
            conv=ndimage.gaussian_filter1d(yrs,sigma=convKernelSigmaSample)

        return conv

    def detectThetaEpochs(self, theta_delta_ratio, theta_delta_ratio_threshold=2,theta_epoch_min_length_ms = 500,sampling_rate=20000):
        """
        Detect theta epochs from the thetaDelta ratio

        Will identify epochs for which thetaDeltaRatio is above thetaDeltaRatioThreshold
        Epochs that are shorter than thetaEpochMinLengthMs will be removed.

        Arguments
        theta_delta_ratio: 1D numpy array containing the thetaDeltaRatio
        theta_delta_ratio_threshold: threshold above which we are in a theta epoch
        theta_epoch_min_length_ms: minimal duration of a theta epochs in ms
        """
        theta_epoch_min_length_samples = sampling_rate/1000 * theta_epoch_min_length_ms
        above_threshold = (theta_delta_ratio > theta_delta_ratio_threshold).astype(int)

        x = np.diff(above_threshold)

        start = np.where(x>0)[0]
        end = np.where(x<0)[0]

        if start.shape[0]==0 and end.shape[0]==0:
            epochs=np.array([])
            return epochs
        else:
            if start.shape[0] == 0:
                if above_threshold[0]==1: # if there is no start but data above threshold
                    start = np.array([0])

            if end.shape[0] == 0:
                if above_threshold[0]==1: # if there is no end but data above threshold
                    end = np.array([theta_delta_ratio.shape[0]-1])

            if end[0] < start[0]: # started with an end
                start = np.concatenate([np.array([0]),start])

            if end[-1] < start[-1]: # end with a start
                end = np.concatenate([end, np.array([theta_delta_ratio.shape[0]-1])])

            if start.shape[0] != end.shape[0]:
                print("problem with start and end of epochs")

            epochs = np.vstack([start,end]).T

            # check that the epocs have the minimal length
            epochs = epochs[(epochs[:,1]-epochs[:,0])> theta_epoch_min_length_samples]

            return epochs 

    def detect_cycles(self, filteredData):
        """
        Detecte the beginning and end of theta cycle

        The beginning and end is the positive to negative transition

        Argument:
        filteredData: 1D numpy array of a filtered signal with a mean near 0.

        Return:
        cycles: 2D array with 2 columns. Each row is the beginning and end of a cycle
        """
        x = np.diff((filteredData>0).astype(int))
        cycleStart = np.where(x<0)[0]
        cycles = np.vstack([cycleStart[:-1],cycleStart[1:]]).T
        return cycles

    def detect_cycles_in_epochs(self, filteredData,epochs):
        """
        Detect individual cycles within epochs with a given oscillation

        Arguments:
        filteredData: 1D numpy array with the filtered signal
        epochs: 2D numpy array with 2 columns, each row is the beginning and end of an epoch
        """
        myCycleList = []
        for i in range(epochs.shape[0]):
            start = epochs[i,0]
            end = epochs[i,1]
            epoch_data = filteredData[start:end]
            cycles = self.detect_cycles(epoch_data)+start
            myCycleList.append(cycles)
        if myCycleList:
            allCycles = np.concatenate(myCycleList)
        else:
            allCycles = myCycleList
        return allCycles

    def detect_theta_cycles_one_channel(self, data):
        """
        Function to get the theta cycles of a signal from an electrode

        Arguments:
        data: 1D numpy array with the signal in which to detect theta cycles
        
        The parameters for the detection are set in the init() for the Theta class
        
        Returns
        A tuple with 
        - 2D array with theta epochs, 2 columns, each row is the start and end of an epoch
        - 2D array with individual theta cycles, 2 columns, each row is the start and end of a cycle
        """
        thetaPow = self.calculate_oscillation_power(data, 
                                                    min_frequency=self.min_theta_frequency, 
                                                    max_frequency=self.max_theta_frequency, 
                                                    sampling_rate=self.session.sampling_rate,
                                                    order=self.order,
                                                    conv_kernel_sigma_ms=self.conv_kernel_sigma_ms)
        deltaPow = self.calculate_oscillation_power(data, 
                                                    min_frequency=self.min_delta_frequency, 
                                                    max_frequency=self.max_delta_frequency, 
                                                    sampling_rate=self.session.sampling_rate,
                                                    order=self.order,
                                                    conv_kernel_sigma_ms=self.conv_kernel_sigma_ms)
        tdr = thetaPow/deltaPow   


        epochs = self.detectThetaEpochs(theta_delta_ratio=tdr, 
                               theta_delta_ratio_threshold=self.theta_delta_ratio_threshold,
                               theta_epoch_min_length_ms = self.theta_epoch_min_length_ms,
                               sampling_rate=self.session.sampling_rate)

        filteredData = self.butter_bandpass_filter(data = data, 
                           lowcut=self.min_theta_frequency, 
                           highcut=self.max_theta_frequency, 
                           fs=self.session.sampling_rate, 
                           order=self.order)

        cycles = self.detect_cycles_in_epochs(filteredData,epochs)

        return epochs, cycles

    def detect_theta_cycles_one_session(self, channel_list=None):
        """
        Detect theta epochs and cycles for a recording session. 

        The epochs and cycles will be detected for each channel of the channel_list
        
        The parameters for theta detection are set in the init() of the Theta class

        We will read the data into RAM one file at a time to prevent filling up the RAM. 
        Only the data from the channels in the channel list will be loaded into memory.

        We create a epochs and cycles dictionaries. The keys of the dictionary are the channel numbers. 
        In epochs[0] you have a 2D array with one epoch per row. This comes from the detection on channel 0.
        In cycles[3] you have a 2D array with one cycle per row. This comes from the detection on channel 3.

        The epochs and cycles dictionaries are saved as a pickle. This way we can load the results from file without redoing this analysis.

        Arguments:
        channel_list: list of channel for which you want the theta epochs and theta cycles. By default all channels but the last one will be processed

        Returns:
        creates self.epochs and self.cycles
        Each is a dictionary with the channel number as key. 
        For example, cycles[3] you have a 2D array containint the cycles detected on channel 3, one cycle per row

        """
        if "dat" not in self.session.file_names :
            self.session.load_parameters_from_files()
            
        if channel_list is None:
            channel_list = list(range(self.session.n_channels-1))

        df = Dat_file_reader(file_names=self.session.file_names["dat"], n_channels=self.session.n_channels)


        epochs = {} # to store the data for all channels
        cycles = {} # to store the data for all channels

        for c in channel_list:
            epochs[c]=[]
            cycles[c]=[]


        for i in range(len(self.session.file_names["dat"])):
            print("reading from {}".format(df.file_names[i]))

            data = df.get_data_one_block(start_sample=df.files_first_sample[i], 
                                      end_sample=df.files_last_sample[i],
                                      channels=np.asarray(channel_list))
            for j,c in tqdm(enumerate(channel_list)):
                ep, cy = self.detect_theta_cycles_one_channel(data[j])
                epochs[c].append(ep+df.files_first_sample[i])
                cycles[c].append(cy+df.files_first_sample[i])


        for c in channel_list: # create one array per channel, and set the time in seconds
            epochs[c]=np.concatenate(epochs[c])/self.session.sampling_rate
            cycles[c]=np.concatenate(cycles[c])/self.session.sampling_rate


        file_name = self.session.fileBase+".theta_detection.pkl"
        print("Saving results in",file_name)
        pickle.dump((epochs, cycles),open(file_name,"wb"))

        self.epochs = epochs
        self.cycles = cycles

        return 

    def load_theta_epochs_and_cycles(self):
        """
        Function to load the theta epochs and theta cycles from file
        
        The data that will be loaded was created by detect_theta_cycles_one_session()
        
        """
        file_name = self.session.fileBase+".theta_detection.pkl"
        if not path.isfile(file_name):
            raise ValueError("{} not found. Try running Theta.detect_theta_cycles_one_session() first.".format(file_name))        
        
        self.epochs,self.cycles = pickle.load( open(file_name, "rb") )