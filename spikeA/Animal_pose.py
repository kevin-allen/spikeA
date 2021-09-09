import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy import ndimage
import os.path

from spikeA.Dat_file_reader import Dat_file_reader
from spikeA.ttl import detectTTL
from spikeA.Intervals import Intervals
from spikeA.Session import Session
class Animal_pose:
    """
    Class containing information about the pose (position and orientation) of an animal in time
    
    The position is in x,y,z and the orientation in Euler angles (yaw,pitch,roll). 
    When not available, data are set to np.NAN
    
    To ease computations, the time intervals between data point is kept constant.
    
    Position data are in cm.
    Angular data are in radians to make computations simpler
    
    Attributes:
    
        pose: 2D numpy array, columns are (time, x,y,z,yaw,pitch,roll). This is a pointer to pose_ori or pose_inter
        pose_ori: 2D numpy array of the original data loaded
        pose_inter: 2D numpy array of the pose data that are within the intervals set
        inter: Interval object
        occupancy_cm_per_bin: cm per bin in the occupancy map
        occupancy_map: 2D numpy array containing the occupancy map
        occupancy_bins: list of 2 x 1D array containing the bin edges to create the occupancy map (used when calling np.histogram2d)
        occupancy_smoothing: boolean indicating of the occupancy map was smoothed
        smoothing_sigma_cm: standard deviation in cm of the gaussian smoothing kernel used to smooth the occupancy map.
        ttl_ups: list of 1D numpy array containing the sample number in the dat files at which a up TTL was detected. This is assigned in pose_from_positrack_files()
        
        hd_occupancy_deg_per_bin: cm per bin in the occupancy map
        hd_occupancy_histogram: 1D numpy array containing the HD occupancy histogram
        hd_occupancy_bins: 1D array containing the bin edges to create the hd occupancy histogram (used when calling np.histogram)
        hd_occupancy_smoothing: boolean indicating of the occupancy map was smoothed
        hd_smoothing_sigma_deg: standard deviation in cm of the gaussian smoothing kernel used to smooth the head-direction occupancy histogram.
        
        
    Methods:
        pose_from_positrack_files()
        save_pose_to_file()
        load_pose_from_file()
        set_intervals()
        unset_intervals()
        occupancy_map()
        head_direction_occupancy_histogram()
        
    """
    def __init__(self, ses=None):
        """
        Constructor of the Animal_pose class
        """
        
        # self.ses can be set here or when loading data from files
        if ses is None:
            self.ses = None
        else :
            if not (issubclass(type(ses),Session)):   ### I needed to change Session to Tetrode_session here to make it run
                raise TypeError("ses should be a subclass of the Session class but is {}".format(type(ses)))
            self.ses = ses # information regarding our session
            
        self.pose = None
        self.pose_ori = None
        self.pose_inter = None
        self.intervals = None
        self.occupancy_cm_per_bin = None
        self.occupancy_map = None
        self.occupancy_bins = None
        self.occupancy_smoothing = None
        self.smoothing_sigma_cm = None
        self.pose_file_extension = ".pose.npy"
    
    def save_pose_to_file(self,file_name=None):
        """
        Save the original pose for this session into an npy file
        
        This is used so we don't have to look at the synchronization of the position data and ephys every time we want the position data.
        
        Arguments
        file_name: If you want to save to a specific file name, set this argument. Otherwise, the self.ses object will be used to determine the file name.
        """
        if self.pose is None:
            raise ValueError("the pose should be set before saving it to file")
        if file_name is None and self.ses is None:
            raise ValueError("self.ses is not set and no file name is given")
        
        if file_name is not None:
            fn = file_name
        else:
            fn = self.ses.fileBase+self.pose_file_extension
            
        print("Saving original pose to",fn)
        np.save(file = fn, arr = self.pose_ori) 
            
    def load_pose_from_file(self,file_name=None):
        """
        Load the pose data from file.
        
        Arguments
        file_name: If you want to save to a specific file name, set this argument. Otherwise, the self.ses object will be used to determine the file name.
        """
        if file_name is None and self.ses is None:
            raise ValueError("self.ses is not set and no file name is given")
        
        if file_name is not None:
            fn = file_name
        else:
            fn = self.ses.fileBase+self.pose_file_extension
        
        
        if not os.path.exists(fn):
            raise OSError(fn+" is missing")
        print("Loading original pose from",fn)
        self.pose_ori = np.load(file = fn) 
        self.pose = self.pose_ori
    
        ## create intervals that cover all the data in self.pose
        if self.intervals is not None:
            # set default time intervals from 0 to the last sample
            self.set_intervals(inter=np.array([[0,self.pose[:,0].max()+1]]))
        else :
             # get intervals for the first time
            self.intervals = Intervals(inter=np.array([[0,self.pose[:,0].max()+1]]))
    
    def set_intervals(self,inter):
        """
        Function to limit the analysis to poses within a set of set specific time intervals
        
        Arguments:
        inter: 2D numpy array, one interval per row, time in seconds
        
        Return:
        The function will set self.intervals to the values of inter
        """
        
        if self.pose is None:
            raise ValueError("the pose should be set before setting the intervals")
        
        self.intervals.set_inter(inter)
        
        # only use the poses that are within the intervals
        self.pose_inter = self.pose_ori[self.intervals.is_within_intervals(self.pose_ori[:,0])] 
        # self.st is now pointing to self.st_inter
        self.pose = self.pose_inter
        #print("Number of poses: {}".format(self.pose.shape[0]))
    
    def unset_intervals(self):
        """
        Function to remove the previously set intervals. 
        
        After calling this function, all poses of the original data loaded will be considered.
        The default interval that includes all poses is set.
        """
        if self.pose is None:
            raise ValueError("pose should be set before setting the intervals")
        
        self.pose = self.pose_ori
        # set default time intervals from 0 to just after the last spike
        self.intervals.set_inter(inter=np.array([[0,self.pose[:,0].max()+1]]))
        #print("Number of poses: {}".format(self.pose.shape[0]))
        
    def head_direction_occupancy_histogram(self, deg_per_bin=10, smoothing_sigma_deg=10, smoothing = True, zero_to_nan = True):
        """
        Function to calculate an head-direction occupancy histogram for head-direction data. 
        The occupancy histogram is a 1D array covering the entire 0-360 degree.
        Each bin of the array contains the time in seconds that the animal spent in the bin.
        The head-direction occupancy histogram is used to calculate head-direction firing rate histograms
        
        The calculations are all done in radians. Bin size (deg_per_bin) and smoothing (smoothing_sigma_deg) are provided in degrees as people are generally more familiar with degrees. The outputs are all in radians
        
        Arguments
        cm_per_deg: deg per bins in the occupancy histogram
        smoothing_sigma_deg: standard deviation of the gaussian kernel used to smooth the occupancy histogram
        smoothing: boolean indicating whether or not smoothing should be applied to the occupancy histogram
        zero_to_nan: boolean indicating if occupancy bins with a time of zero should be set to np.nan
        
        Return
        self.hd_occupancy_histogram and self.hd_occupancy_histogram are set. They are 1D numpy arrays containing the time spent in seconds in a set of head-direction and the edges of the histogram bins (in radians)
        """
        if self.pose is None:
            raise TypeError("Set the self.pose array before attempting to calculate the hd_occupancy_histogram")
        
        # we save this for later use when calculating firing rate maps
        
        self.hd_occupancy_deg_per_bin = deg_per_bin
        self.hd_occupancy_smoothing = smoothing
        self.hd_smoothing_sigma_deg = smoothing_sigma_deg
        
        # remove invalid head direction data, self.pose
        invalid = np.isnan(self.pose[:,4])
        #print("{} invalid rows out of {}, % invalid: {:.2f}".format(invalid.sum(),invalid.shape[0],invalid.sum()/invalid.shape[0]*100 ))
        val = self.pose[~invalid,4]
        
        # calculate the hd occupancy histogram
        self.hd_occupancy_bins = np.arange(-np.pi,np.pi+self.hd_occupancy_deg_per_bin/360*2*np.pi,self.hd_occupancy_deg_per_bin/360*2*np.pi)
        occ,edges = np.histogram(val,bins= self.hd_occupancy_bins)
        
        # calculate the time per sample
        sec_per_sample = self.pose[1,0]-self.pose[0,0] # all rows have equal time intervals between them, we get the first one
                
        # get the time in seconds
        occ = occ*sec_per_sample
        
        # smoothin of occupancy map
        if smoothing:
            occ_sm = ndimage.gaussian_filter1d(occ,sigma=smoothing_sigma_deg/deg_per_bin)
        else:
            occ_sm = occ # if no smoothing just get a reference to occ, because we want to use occ_sm for the rest of the function
          
        # set bins at 0 to np.nan
        if zero_to_nan:
            occ_sm[occ==0] = np.nan
    
        # save the occupancy map for later use
        self.hd_occupancy_histogram = occ_sm
        
    
        
    def occupancy_map_2d(self, cm_per_bin =2, smoothing_sigma_cm = 2, smoothing = True, zero_to_nan = True):
        """
        Function to calculate an occupancy map for x and y position data. 
        The occupancy map is a 2D array covering the entire environment explored by the animal.
        Each bin of the array contains the time in seconds that the animal spent in the bin.
        The occupancy map is used to calculate firing rate maps
        
        Arguments
        cm_per_bin: cm per bins in the occupancy map
        smoothing_sigma_cm: standard deviation of the gaussian kernel used to smooth the occupancy map
        smoothing: boolean indicating whether or not smoothing should be applied to the occupancy map
        zero_to_nan: boolean indicating if occupancy bins with a time of zero should be set to np.nan
        
        Return
        self.occupancy_map is set. It is a 2D numpy array containing the time spent in seconds in a set of bins covering the environment
        """
        
        if self.pose is None:
            raise TypeError("Set the self.pose array before attempting to calculate the occupancy_map_2d")
        
        # we save this for later use when calculating firing rate maps
        self.occupancy_cm_per_bin=cm_per_bin
        self.occupancy_smoothing = smoothing
        self.smoothing_sigma_cm = smoothing_sigma_cm
        
        # remove invalid position data
        invalid = np.isnan(self.pose[:,1:3]).any(axis=1)
        #print("{} invalid rows out of {}, % invalid: {:.2f}".format(invalid.sum(),invalid.shape[0],invalid.sum()/invalid.shape[0]*100 ))
        val = self.pose[~invalid,1:3]
        
        ## determine the size of the occupancy map with the maximum x and y values
        #print("max x and y values: {}".format(val.max(axis=0)))
        xy_max = np.ceil(val.max(axis=0))+cm_per_bin
        #print("max x and y for the np.arange function : {}".format(xy_max))

        # create two arrays that will be our bin edges in the histogram function
        self.occupancy_bins = [np.arange(0,xy_max[0],cm_per_bin),
                               np.arange(0,xy_max[1],cm_per_bin)]
        
        
        # calculate the occupancy map
        occ,x_edges,y_edges = np.histogram2d(x = val[:,0], y= val[:,1],
                                            bins= self.occupancy_bins)
        
        # calculate the time per sample
        sec_per_sample = self.pose[1,0]-self.pose[0,0] # all rows have equal time intervals between them, we get the first one
        #print("time per sample: {}".format(sec_per_sample))
        
        # get the time in seconds
        occ = occ*sec_per_sample
    
        # smoothin of occupancy map
        if smoothing:
            occ_sm = ndimage.filters.gaussian_filter(occ,sigma=smoothing_sigma_cm/cm_per_bin)
        else:
            occ_sm = occ # if no smoothing just get a reference to occ, because we want to use occ_sm for the rest of the function
          
        # set bins at 0 to np.nan
        if zero_to_nan:
            occ_sm[occ==0] = np.nan
    
        # save the occupancy map for later use
        self.occupancy_map = occ_sm
        
        
    

    def pose_from_positrack_files(self,ses=None, ttl_pulse_channel=None, interpolation_frequency_hz = 50, extension= "positrack2"):


        """
        Method to calculute pose at fixed interval from a positrack file.
        
        Arguments
        ses: A Session object
        ttl_pulse_channel: channel on which the ttl pulses were recorded. If not provided, the last channel is assumed
        interpolation_frequency_hz: frequency at which with do the interpolation of the animal position
        extension: file extension of the file with position data (positrack or positrack2)
                
        Return
        No value is returned but self.time and self.pose are set
        
        """
        
        if ses is None and self.ses is None:
            raise TypeError("Please provide a session object with the ses argument")
        
        if ses is not None:
            if not (issubclass(type(ses),Session) or isinstance(ses,Session)): 
                raise TypeError("ses should be a subclass of the Session class")
            self.ses = ses # update what is in the self.ses
        
        # we loop for each trial, check the syncrhonization, append the position data and ttl time 
        # (taking into account the length of previous files)

        # counter for trial offset from beginning of recordings
        trial_sample_offset = 0
        # list to store the position array of each trial
        posi_list = []
        # list to store the up values (including the offset)
        self.ttl_ups = []
        
        # interpolate to have data for the entire .dat file (padded with np.nan at the beginning and end when there was no tracking)
        interpolation_step = self.ses.sampling_rate/interpolation_frequency_hz # interpolate at x Hz
        print("Interpolation step: {} samples".format(interpolation_step))


        #loop for trials
        for i,t in enumerate(self.ses.trial_names):
            dat_file_name = self.ses.path + "/" + t+".dat"
            positrack_file_name = self.ses.path + "/" + t+"."+ extension
            print(dat_file_name)
            print(positrack_file_name)

            positrack_file = Path(positrack_file_name)
            
            if not positrack_file.exists() :
                raise OSError("positrack file {} missing".format(positrack_file_name))

            # read the ttl channel for positrack
            df = Dat_file_reader(file_names=[dat_file_name],n_channels = self.ses.n_channels)
            ttl_channel_data = df.get_data_one_block(0,df.files_last_sample[-1],np.array([self.ses.n_channels-1]))
            ttl,downs = detectTTL(ttl_data = ttl_channel_data)
            print("Number of ttl pulses detected: {}".format(ttl.shape[0]))

            # read the positrack file
            if extension=="positrack" :
                pt = pd.read_csv(positrack_file_name, delimiter=" ")
                    
            elif extension=="positrack2":
                pt = pd.read_csv(positrack_file_name)
            elif extension=="trk":
                data = np.reshape(np.fromfile(file=positrack_file_name,dtype=np.int32),(-1,21))
                data = data.astype(np.float32)
                pt = pd.DataFrame({"x":data[:,11], "y":data[:,12],"hd": data[:,10]})
            else :
                raise ValueError("extension not supported")
  
            print("Number of lines in positrack file: {}".format(pt.shape[0]))
            if ttl.shape[0] != pt.shape[0]:
                print("alignment problem")

                # we will need code to solve simple problems 
                #
                #
                raise ValueError("Synchronization problem (positrack {} and ttl pulses {}) for trial {}".format(pt.shape[0],ttl.shape[0],t))

            # create a numpy array with the position data
            d = np.stack([pt["x"].values,pt["y"].values,pt["hd"].values]).T 
            # set invalid values to np.nan
            d[d==-1.]=np.nan

            # we want to treat hd values as cos and sin for interpolation
            c = np.cos(d[:,2]/180*np.pi)
            s = np.sin(d[:,2]/180*np.pi)
            # add cos and sin to our d array
            x = np.stack([c,s]).T
            d = np.hstack([d,x])
            print(d.shape)

            valid = np.sum(~np.isnan(d))
            invalid = np.sum(np.isnan(d))
            prop_invalid = invalid/d.size
            print("Invalid values: {}".format(invalid))
            print("Valid values: {}".format(valid))
            print("Percentage of invalid values: {:.3}%".format(prop_invalid*100))
            if prop_invalid > 0.05:
                print("****************************************************************************************")
                print("WARNING")
                print("The percentage of invalid values is very high. The quality of your data is compromised.")
                print("Solve this problem before continuing your experiments.")
                print("****************************************************************************************")  

            # estimate functions to interpolate
            fx = interp1d(ttl[:], d[:,0], bounds_error=False) # x we will start at 0 until the end of the file
            fy = interp1d(ttl[:], d[:,1], bounds_error=False) # y 
            fhdc = interp1d(ttl[:], d[:,3], bounds_error=False) # cos
            fhds = interp1d(ttl[:], d[:,4], bounds_error=False) # sin

            # set the time points at which we want a position
            new_time = np.arange(0, df.total_samples,interpolation_step)

            # interpolate
            new_x = fx(new_time)
            new_y = fy(new_time)
            new_hdc = fhdc(new_time)
            new_hds = fhds(new_time)

            # one array with all the interpolated data
            posi_d = np.vstack([new_time+trial_sample_offset, # time in sample from the beginning of first trial (0)
                                new_x,new_y,new_hdc,new_hds]).T # interpolated position data

            # store the data in a lists of arrays
            posi_list.append(posi_d)
            
            # store the ttl up
            self.ttl_ups.append(ttl[:]+trial_sample_offset)

            # change the offset for the next trial
            trial_sample_offset+=df.total_samples

        # put all the trials together
        posi = np.concatenate(posi_list)
        print("shape of position data for all trials:",posi.shape)

        # if we have more than 1 trial, we need to re-interpolate so that we constant time difference between position data
        #if ses.n_trials > 1:

        # estimate functions to interpolate
        fx = interp1d(posi[:,0], posi[:,1], bounds_error=False) # x we will start at 0 until the end of the file
        fy = interp1d(posi[:,0], posi[:,2], bounds_error=False) # y 
        fhdc = interp1d(posi[:,0], posi[:,3], bounds_error=False) # cos
        fhds = interp1d(posi[:,0], posi[:,4], bounds_error=False) # sin

        # new time to interpolate
        nt = np.arange(0, posi[-1,0]+interpolation_step,interpolation_step)

        #
        new_x = fx(nt)
        new_y = fy(nt)
        new_hdc = fhdc(nt)
        new_hds = fhds(nt)

        # get back the angle from the cosin and sin
        new_hd = np.arctan2(new_hdc,new_hds) # this does not work at the moment.

        # contain time, x,y,z, yaw, pitch, roll
        # index:    0   1,2,3, 4,    5,     6
        self.pose_ori = np.empty((new_x.shape[0],7),float) # create the memory
        self.pose = self.pose_ori # self.pose points to the same memory as self.pose_ori
        
        self.pose[:] = np.nan
        self.pose[:,0] = nt/self.ses.sampling_rate # from sample number to time in seconds
        self.pose[:,1] = new_x/self.ses.px_per_cm # transform to cm
        self.pose[:,2] = new_y/self.ses.px_per_cm # transform to cm
        self.pose[:,4] = new_hd
        
        ## create intervals that cover the entire session
        if self.intervals is not None:
            # set default time intervals from 0 to the last sample
            self.set_intervals(inter=np.array([[0,self.pose[:,0].max()+1]]))
        else :
             # get intervals for the first time
            self.intervals = Intervals(inter=np.array([[0,self.pose[:,0].max()+1]]))