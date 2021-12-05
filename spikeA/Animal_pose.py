import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
import os.path
import os
import spikeA.spatial_properties

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
    Yaw should be from -180 to 180, 0 is in direction of positive x-axis, pi/2 up, -pi/2 down.
    
    Attributes:
    
        pose: 2D numpy array, columns are (time, x,y,z,yaw,pitch,roll). This is a pointer to pose_ori or pose_inter
        pose_ori: 2D numpy array of the original data loaded, This array should never be modified!
        pose_inter: 2D numpy array of the pose data that are within the intervals set
        pose_rolled: 2D numpy array of the pose data but shuffled (or rolled) to get shuffling distributions
        speed: 1D numpy array of speed data of the original pose data
        intervals: Interval object
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
        save_pose_to_file()
        load_pose_from_file()
        pose_from_positrack_files()
        percentage_valid_data()
        set_intervals()
        unset_intervals()
        occupancy_map()
        head_direction_occupancy_histogram()
        speed_from_pose()
        roll_pose_over_time()
        
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
        self.speed = None
        self.pose_file_extension = ".pose.npy"
    
    def roll_pose_over_time(self,min_roll_sec=20):
        """
        Function to roll the spatial data (self.pose[:,1:7]) relative to the time (self.pose[0,:]).
        
        This function is used to "shuffle" the position data relative to the spike train of neurons in order to get maps that would be expected if the neurons was not spatially selective.
        This procedure is used to calculated significance thresholds for spatial information score, grid scores, etc.
        The position data are shifted forward from their current time by a random amount that is larger than min_roll_sec. 
        You should set your intervals before calling this function.
        
        This function will set self.pose to self.pose_rolled
        It is recommended to make a copy of pose before the shuffling procedure and reset ap.pose to the original copy when you are done.
        
        
        Example:
        
        # keep a copy of the pose that we started with
        pose_at_start = sp.ap.pose.copy()
        
        # allocate memory for the shuffle data
        sp.head_direction_shuffle=np.empty(iterations)
        
        
        for i in range(iterations):
            sp.ap.roll_pose_over_time() # shuffle the position data 
            sp.firing_rate_head_direction_histogram(deg_per_bin=deg_per_bin, smoothing_sigma_deg = smoothing_sigma_deg, smoothing=smoothing)  
            sp.head_direction_shuffle[i] = sp.head_direction_score()[2] # calculate the new HD score (vector length only) with the shuffled HD data
            
            sp.ap.pose=pose_at_start # reset the pose to the one we started with

        # calculate the threshold
        sp.head_direction_score_threshold =  np.percentile(sp.head_direction_shuffle,percentile)
        
        
        
        """
        
        total_time_sec = self.intervals.total_interval_duration_seconds()
        if total_time_sec < 2*min_roll_sec:
            raise valueError("total time in intervals should be larger than 2*min_roll_sec")
        
        # random time shift between what is possible
        time_shift = np.random.default_rng().uniform(min_roll_sec,total_time_sec-min_roll_sec,1)
        time_per_datapoint = self.pose[1,0]-self.pose[0,0]
        shift=int(time_shift/time_per_datapoint)
        
        self.pose_rolled = np.empty(self.pose.shape)
        
        self.pose_rolled[:,0] = self.pose[:,0] # time does not roll
        self.pose_rolled[:,1:7] = np.roll(self.pose[:,1:7],shift=shift,axis=0)
        
        if self.pose.shape[1] > 7:
            self.pose_rolled[:,7:] = self.pose[:,7:]
        
        
        self.pose = self.pose_rolled
        
    
    
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
        
        The original pose data from the file are stored in self.pose_ori
        When we set intervals, self.pose points to self.pose_inter. self.pose_inter can be modified as we want.
        
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
        #print("Loading original pose from",fn)
        self.pose_ori = np.load(file = fn) 
        self.pose = self.pose_ori.copy() # the self.pose should not point to the self.pose_ori
    
        ## create intervals that cover all the data in self.pose
        if self.intervals is not None:
            # set default time intervals from 0 to the last sample
            self.set_intervals(inter=np.array([[0,self.pose[:,0].max()+1]]))
        else :
             # get intervals for the first time
            self.intervals = Intervals(inter=np.array([[0,self.pose[:,0].max()+1]]))
            self.set_intervals(inter=np.array([[0,self.pose[:,0].max()+1]]))
            
    def percentage_valid_data(self,columnIndex=1):
        """
        Function to return the percentage of valid data point in the .pose array
        
        Arguments:
        columnIndex: column index to use in the pose array to calculate the proportion of valid data points
        
        Return: Percentage of valid data point in .pose
        """
        return np.sum(~np.isnan(self.pose[:,columnIndex]))/self.pose.shape[0]*100
        
        
        
    def set_intervals(self,inter,timeColumnIndex=0):
        """
        Function to limit the analysis to poses within a set of set specific time intervals.
        
        Each time it is called, it starts from the data in self.pose_ori, which is the data that was loaded from file and never change.
        
        Arguments:
        inter: 2D numpy array, one interval per row, time in seconds
        timeColumnIndex: index of the column containing the time in the pose matrix
        
        Return:
        The function will set self.intervals to the values of inter
        """
        
        if self.pose is None:
            raise ValueError("the pose should be set before setting the intervals")
        
        self.intervals.set_inter(inter)
        
        # only use the poses that are within the intervals
        # always start over from the original pose that should never change
        self.pose_inter = self.pose_ori[self.intervals.is_within_intervals(self.pose_ori[:,timeColumnIndex])] # this should create a copy of self.pose_ori, not a reference
        
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
        
        self.pose = self.pose_ori.copy() # create a copy of our pose_ori, not a pointer

        # set default time intervals from 0 to just after the last spike
        self.intervals.set_inter(inter=np.array([[0,self.pose[:,0].max()+1]]))
        #print("Number of poses: {}".format(self.pose.shape[0]))
        
    def head_direction_occupancy_histogram(self, deg_per_bin=10, smoothing_sigma_deg=10, smoothing = True, zero_to_nan = True):
        """
        Function to calculate an head-direction occupancy histogram for head-direction data. 
        The occupancy histogram is a 1D array covering the entire 0-360 degree.
        Each bin of the array contains the time in seconds that the animal spent in the bin.
        The head-direction occupancy histogram is used to calculate head-direction firing rate histograms
        
        The calculations are all done in radians [-np.pi,np.pi]. Bin size (deg_per_bin) and smoothing (smoothing_sigma_deg) are provided in degrees as people are generally more familiar with degrees. The outputs are all in radians
        
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
            occ_sm = ndimage.gaussian_filter1d(occ,sigma=smoothing_sigma_deg/deg_per_bin,mode="wrap")
        else:
            occ_sm = occ # if no smoothing just get a reference to occ, because we want to use occ_sm for the rest of the function
          
        # set bins at 0 to np.nan
        if zero_to_nan:
            occ_sm[occ==0] = np.nan
    
        # save the occupancy map for later use
        self.hd_occupancy_histogram = occ_sm
        
    
        
    def occupancy_map_2d(self, cm_per_bin =2, smoothing_sigma_cm = 2, smoothing = True, zero_to_nan = True,
                        xy_range=None):
        """
        Function to calculate an occupancy map for x and y position data. 
        The occupancy map is a 2D array covering the entire environment explored by the animal.
        Each bin of the array contains the time in seconds that the animal spent in the bin.
        The occupancy map is used to calculate firing rate maps
        The x and y data will end up being the rows and columns of the 2D array. The same will apply to the spike position. This is because the first axis of a numpy array is the row and second is column.
        
        
        Arguments
        cm_per_bin: cm per bins in the occupancy map
        smoothing_sigma_cm: standard deviation of the gaussian kernel used to smooth the occupancy map
        smoothing: boolean indicating whether or not smoothing should be applied to the occupancy map
        zero_to_nan: boolean indicating if occupancy bins with a time of zero should be set to np.nan
        xy_range: 2D np.array of size 2x2 [[xmin,ymin],[xmax,ymax]] with the minimal and maximal x and y values that should be in the occupancy map, default is None and the values are calculated from the data.         
        
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
        
        ## determine the size of the occupancy map with the minimum and maximum x and y values
        #print("max x and y values: {}".format(val.max(axis=0)))
        if xy_range is None:
            xy_max = np.ceil(val.max(axis=0))+cm_per_bin
            xy_min = np.floor(val.min(axis=0))-cm_per_bin
        else :
            xy_max= xy_range[1,:]
            xy_min= xy_range[0,:]
        #print("min and max x and y for the np.arange function : {}, {}".format(xy_min,xy_max))

        # create two arrays that will be our bin edges in the histogram function
        self.occupancy_bins = [np.arange(xy_min[0],xy_max[0],cm_per_bin),
                               np.arange(xy_min[1],xy_max[1],cm_per_bin)]
        
        
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
     
    
        
    def occupancy(self, arena):
        """
        Function to calculate the proportions of bins of the occupancy map covered by the animal. Can be used for rectanglular and circular arenas.
        
        This function is very specific to some recording environment. We should try to make it usable irrespective of the code name of the environment.
        
        Arguments
        arena: specifies the shape of the arena:circle/square        
        
        Return
        occupancy
        """
        if not hasattr(self, 'occupancy_map'):
            raise TypeError('You have to call ap.occupancy_map_2d() before calling this function')
        
        if arena == 'square':
            area = self.occupancy_map.shape[0]*self.occupancy_map.shape[1] # area of a rectangle

        elif arena == 'circle':
            # use the smaller dimension as diameter of the circle as there might be reflections outside the arena
            area = ((np.min(self.occupancy_map.shape)/2)**2)*np.pi # area of a circle
            
        else:
            raise TypeError("This arena shape is not supported. Only square or circle can be used.")

        occupancy = self.occupancy_map[~np.isnan(self.occupancy_map)].shape[0]/area
        
        return occupancy
     
        

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
                pt = pd.read_csv(positrack_file_name, delimiter=" ", index_col=False)

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
                # if there are just 1 or 2 ttl pulses missing from positrack, copy the last 1 or 2 lines
                if extension =="positrack" and (ttl.shape[0] == (pt.shape[0]+1) or ttl.shape[0] == (pt.shape[0]+2) or ttl.shape[0] == (pt.shape[0]+3)):
                    original_positrack_file = self.ses.path + "/" + t+"o."+ extension
                    missing = ttl.shape[0]-pt.shape[0]
                    pt_mod = pt.append(pt[(pt.shape[0]-missing):(pt.shape[0]+1)])
                    print("Number of lines in adjusted positrack file:", pt_mod.shape[0])
                    os.rename(positrack_file_name, original_positrack_file)
                    pt_mod.to_csv(positrack_file_name, sep=' ')
                    pt = pt_mod
                    print("Alignment problem solved by adding up to 3 ttl pulses to positrack")
                elif extension=="positrack" and (ttl.shape[0]<pt.shape[0]):
                    original_positrack_file = self.ses.path + "/" + t+"o."+ extension
                    pt_mod = pt[:ttl.shape[0]]
                    print("Number of lines in adjusted positrack file:", pt_mod.shape[0])
                    os.rename(positrack_file_name, original_positrack_file)
                    pt = pt_mod
                    pt.to_csv(positrack_file_name, sep=' ')
                    print("Alignment problem solved by deleting superfluent ttl pulses in positrack")

                # we will need more code to solve simple problems 
                #
                #
                else:
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
            
            
    def speed_from_pose(self, sigma=1):
        """
        Method to calculute the speed (in cm/s) of the animal from the position data
        The speed at index x is calculated from the distance between position x and x+1 and the sampling rate.
        Then a Gaussian kernel is applied for smoothing.
        
        Arguments
        sigma: for Gaussian kernel
                
        Return
        No value is returned but self.speed is set
        
        """
        if self.pose is None:
            raise TypeError("Set the self.pose array before attempting to calculate the speed")
        
        # create empty speed array
        self.speed = np.empty((len(self.pose[:,0]),1),float)
        
        
        # calculate the time per sample
        sec_per_sample = self.pose[1,0]-self.pose[0,0] # all rows have equal time intervals between them, we get the first one
        
        # calculate the distance covered between the position data and divide by time per sample
        distance = np.diff(self.pose, axis=0, append=np.nan)
        speed = np.sqrt(distance[:,1]**2 + distance[:,2]**2)/sec_per_sample
        
        # apply gaussian filter for smoothing
        self.speed = gaussian_filter1d(speed, sigma=sigma)
    
    
    def detect_border_pixels_in_occupancy_map(self):
        """
        Method to detect the border pixels in an occupancy map
        
        Make sure you set the xy-range for the occupancy map. If the arena borders and the map borders touch, the border detection will not work properly.
        
        A border map is created in which all pixels are 0 and border pixels are 1.
        
        This function in implemented in c code located in spatial_properties.c, spatial_properties.h and _spatial_properties.pyx
        
        No arguments:
        
        Returns:
        2D numpy array containing with the border pixels set to 1 and the rest set to 0
        
        Example
        
        from spikeA.Animal_pose import Animal_pose
        import numpy as np
        import matplotlib.pyplot as plt
        
        # create a circular environment
        points = np.arange(-20,20,1)
        xs,ys = np.meshgrid(points,points)
        occ_map = np.sqrt(xs**2+ys**2)
        occ_map[occ_map>10] = -1.0
        
        ap = Animal_pose()
        ap.occupancy_map=occ_map
        brd = ap.detect_border_pixels_in_occupancy_map()
        
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(occ_map)
        ax[1].imshow(brd)
        plt.show()
        """
        
        # check for occupancy map
        if not hasattr(self, 'occupancy_map'):
            raise TypeError("You need to call ap.occupancy_map_2d(xy_range=np.array([[...,...],[...,...]])) before calling ap.detect_border_pixels_in_occupancy_map()")
        
        ## convert nan values to -1 for C function
        occ_map = self.occupancy_map.copy()
        
        occ_map[np.isnan(occ_map)]=-1.0
        
        ## create an empty array of the appropriate dimensions to store the border pixels
        border_map = np.zeros_like(occ_map,dtype=np.int32)
        spikeA.spatial_properties.detect_border_pixels_in_occupancy_map_func(occ_map,border_map)
        return border_map                                 

    
    def invalid_outside_spatial_area(self, shape=None, radius=None, length=None, center=None):
        """
        Method that set the position data (self.pose[:,1:7]) outside a defined zone to np.nan.
        
        The area can be a circle or a square.
        
        To undo, call self.unset_intervals() or self.set_intervals(). unset_intervals() and set_intervals() use the data stored in self.ori_pose
        
        This function should be called **after** setting any relevant Intervals for the analysis.
        
        Arguments:
        shape: "circle","square"
        radius: radius of a circle, only needed if working with circle
        length: length of a square
        center: 1D np.array of size 2, [x,y], center of a circle/square
        
        Return:
        Nothing is returned. self.pose[,1:7] are set to np.nan if the animal is not in the zone.
        """
        valid_shapes = ["circle","square"]
        
        if not shape in valid_shapes:
            raise ValueError("shape should be part of the list {}".format(valid_shapes))
        
        # if center is not specified, determine from center of mass of the occupancy map
        if center is None:
            self.occupancy_map_2d()
            occ_map = self.occupancy_map.copy()
            # nan values need to be 0
            occ_map[np.isnan(occ_map)]=0.0
            # the occupancy map is rotated by -90° to the pose date. We need to rotate back.
            occ_map=ndimage.rotate(occ_map, 90)
            # get the center of mass of the occupancy map
            center_occ=ndimage.measurements.center_of_mass(occ_map)
            #print(center_occ)
            # we need to transform back from the occupancy map to the pose data. 
            # The y axis starts in the top left orner for the occ_map, but in the bottom left corner for the pose data.
            x_range_pose = np.nanmax(self.pose[:,1]-np.nanmin(self.pose[:,1]))
            x_range_occ = occ_map.shape[1]
            y_range_pose = np.nanmax(self.pose[:,2]-np.nanmin(self.pose[:,2]))
            y_range_occ = occ_map.shape[0]
            center = (np.nanmin(self.pose[:,1])+center_occ[1]*x_range_pose/x_range_occ,np.nanmax(self.pose[:,2])-center_occ[0]*y_range_pose/y_range_occ)
            #print(center)

                
        # deal with circle
        if shape == "circle":
            if radius is None:
                raise ValueError("set the radius argument")
            
            # calculate distance to center
            dist = np.sqrt((self.pose[:,1]-center[0])**2 + (self.pose[:,2]-center[1])**2)
            # outside circle = np.nan
            self.pose[dist>radius,1:7] = np.nan
            
        # deal with square
        if shape == "square":
            if length is None:
                raise ValueError("set the length argument")
            
            # set pixels outside square of length length np.nan
            self.pose[self.pose[:,1]>center[0]+length/2,1:7]=np.nan
            self.pose[self.pose[:,2]>center[1]+length/2,1:7]=np.nan
            self.pose[self.pose[:,1]<(center[0]-length/2),1:7]=np.nan
            self.pose[self.pose[:,2]<(center[1]-length/2),1:7]=np.nan


    def invalid_outside_head_direction_range(self, loc = 0, sigma = np.pi/4):
        """
        Method that set the position data (self.pose[:,1:7]) outside a defined head direction range to to np.nan.
        
        To undo, call self.unset_intervals() or self.set_intervals(). unset_intervals() and set_intervals() use the data stored in self.ori_pose
        
        This function should be called **after** setting any relevant Intervals for the analysis.
        
        Arguments:
        loc: angle in radian that will be kept
        sigma: distance from loc (in radians) for which the angles are considered valid
        
        Return:
        Nothing is returned. self.pose[,1:7] are set to np.nan if the animal head direction is not within the set range
        """
        
        
        # angle between loc and HD
        loc_vector = np.array([np.cos(loc)],np.sin(loc))
        hd = self.pose[:,4]
        hdx = np.cos(hd)
        hdy = np.sin(hd)
        
        
        loc_vector=np.array([[np.cos(loc)],[np.sin(loc)]])
    

        hd = self.pose[:,4]
        hdx = np.cos(hd)
        hdy = np.sin(hd)
        hd_vector = np.concatenate([np.expand_dims(hdx,axis=1),np.expand_dims(hdy,axis=1)],axis=1)
        
        delta = np.arccos(hd_vector@loc_vector).flatten()
        self.pose[delta>sigma,1:7] = np.nan
        
        
    def positrack_type(self,ses=None):
        """
        Function trying to answer if the data were collected with positrack or positrack2
        
        Return Type of tracking as a string. Can be "positrack", "positrack2" or "None" 
        """
        if ses is None and self.ses is None:
            raise TypeError("Please provide a session object with the ses argument")
        
        if ses is not None:
            if not (issubclass(type(ses),Session) or isinstance(ses,Session)): 
                raise TypeError("ses should be a subclass of the Session class")
            self.ses = ses # update what is in the self.ses
        
        t = self.ses.trial_names[0]
        positrack_file_name = self.ses.path + "/" + t+".positrack"
        positrack_file = Path(positrack_file_name)
        if positrack_file.exists() :
            return "positrack"
        
        
        positrack_file_name = self.ses.path + "/" + t+".positrack2"
        positrack_file = Path(positrack_file_name)
        if positrack_file.exists() :
            return "positrack2"
        
        return "None"