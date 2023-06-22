import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
import os.path
import os
from scipy.signal import medfilt
import spikeA.spatial_properties

from spikeA.Dat_file_reader import Dat_file_reader
from spikeA.ttl import detectTTL
from spikeA.Intervals import Intervals
from spikeA.Session import Session
import matplotlib.pyplot as plt


# helper functions
def range_pi(x):
    # wrap on interval [-pi,pi)
    return (x + np.pi) % (2*np.pi) - np.pi

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
        distance: 1D numpy array of total distance of the original pose data
        intervals: Interval object
        occupancy_cm_per_bin: cm per bin in the occupancy map
        occupancy_map: 2D numpy array containing the occupancy map
        occupancy_bins: list of 2 x 1D array containing the bin edges to create the occupancy map (used when calling np.histogram2d)
        occupancy_smoothing: boolean indicating of the occupancy map was smoothed
        smoothing_sigma_cm: standard deviation in cm of the gaussian smoothing kernel used to smooth the occupancy map.
        ttl_ups: list of 1D numpy array containing the sample number in the dat files at which a up TTL was detected. This is assigned in pose_from_positrack_files()
        pt_times: list containing the positrack times (positrack2 only)
        
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
    
    def plot_head_direction_heading_correlation(self,ax, min_speed=20):
        """
        Method to plot the head-direction data agains the movement heading
        
        Arguments:
        ax: matplotlib axes on which to plot
        min_speed: float, minimal running speed in cm/sec for the data to be included in the plot
        
        Returns nothing, but draw on the matplotlib axes
        """
        t= self.pose[:,0]
        x= self.pose[:,1]
        y= self.pose[:,2]
        hd= self.pose[:,4]
        
        xd = np.diff(x,append=np.nan)
        yd = np.diff(y,append=np.nan)
        td = np.diff(t,append=np.nan)
        heading = np.arctan2(yd,xd)
        speed= np.sqrt(xd**2+yd**2)/td
        
        # remove data points with abnormally high running speed
        speed[speed>100] = np.nan
        indices = speed >  min_speed # we want to focus on when the animal is moving, when head-direction and heading should be aligned
        
        ax.scatter(heading[indices],hd[indices],s=1,alpha=0.2)
        ax.set_xlabel("Movement heading")
        ax.set_ylabel("Head direction")
    
    
    def test_head_direction_heading_correlation(self,min_speed = 10, max_median_delta=3.1416/3, plot=False, throw_exception = True):
        """
        Method to test whether the head-direction data are aligned with movement heading. This should be the case for normal dataset in which the animal travels in the direction of its head-direction.

        If not, a figure is generated so that the user can visualize the problem and an exception is thrown to prevent the user from working with corrupted position data.

        Arguments:
        
        min_speed: float, minimal speed (cm/sec) that will be included in the analysis, HD is coupled with movement heading mainly when the animal is runing.
        max_median_delta: float, maximal median delta heading-HD, above which the function consider that there is a problem with the data
        plot: boolean, whether to plot the results or now
        throw_exception: boolean, whether to throw and exception if delta heading-HD is larger than max_median_delta
        
        Returns nothing
        If the heading is not aligned with head-direction, the function will plot the results and throw an exception if throw_exception == True
        """
        t= self.pose[:,0]
        x= self.pose[:,1]
        y= self.pose[:,2]
        hd= self.pose[:,4]


        # check if range is larger than 2*pi, if so assumes degrees
        # data coming from self.pose should not be in degrees
        #hdMin, hdMax = np.nanmin(hd),np.nanmax(hd)
        #hdRange = hdMax - hdMin
        #if hdRange > 2*np.pi+0.01: 
        #    print("hd range {}, transforming to radians".format(hdRange))
        #    hd = hd /180 * np.pi # this is 0 to 2*pi
        #    hd[hd>np.pi] = hd[hd>np.pi]-(2*np.pi) # from -pi to pi


        xd = np.diff(x,append=np.nan)
        yd = np.diff(y,append=np.nan)
        td = np.diff(t,append=np.nan)
        heading = np.arctan2(yd,xd)
        speed= np.sqrt(xd**2+yd**2)/td
        
        # remove data points with abnormally high running speed
        speed[speed>100] = np.nan
        indices = speed > min_speed # we want to focus on when the animal is moving, when head-direction and heading should be aligned

        shd = np.sin(hd)
        chd = np.cos(hd)

        # angle between heading and head-direction
        mvVectors = np.stack([xd,yd]) # not unitary vectors
        hdVectors = np.stack([chd,shd]) # already unitary vectors

        mvLength = np.expand_dims(np.linalg.norm(mvVectors,axis=0),axis=0)
        mvVectors = mvVectors/ mvLength # after this are unitary vectors
        delta = np.arccos((np.sum((mvVectors*hdVectors),axis=0))) # angle between vectors (HD and mvH)
        
        medDelta = np.nanmedian(delta[indices]) # median angle for valid indices (where animal is moving)

        if medDelta > max_median_delta or plot:
            fig,axes = plt.subplots(nrows=1, ncols=4,figsize=(8,2), constrained_layout=True) # we use pyplot.subplots to get a figure and axes.    
            axes[0].scatter(x,y)
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")
            axes[0].invert_yaxis()
            axes[1].hist(speed,bins=30)
            axes[1].set_xlabel("Speed (cm/sec)")
            axes[2].scatter(heading[indices],hd[indices],s=1,alpha=0.1)
            axes[2].set_xlabel("Movement heading")
            axes[2].set_ylabel("Head direction")
            axes[3].hist(delta,bins=30)
            axes[3].set_xlabel("Delta MVHD-HD")
            axes[3].set_title("Median delta {:.3f}".format(medDelta))
            plt.show()
        if medDelta > max_median_delta and throw_exception:
            raise ValueError("The movement heading and head-direction of the animal are not aligned.\n The median difference {:.3f} is larger than {:.3f}".format(medDelta,max_median_delta))

    
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
        
    
    
    def save_pose_to_file(self,file_name=None,verbose=True):
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
        
        if verbose:
            print("Saving original pose (shape: {}) to".format(self.pose_ori.shape),fn)
        np.save(file = fn, arr = self.pose_ori) 
            
    def load_pose_from_file(self,file_name=None, verbose=False, pose_file_extension = None):
        """
        Load the pose data from file.
        
        The original pose data from the file are stored in self.pose_ori
        When we set intervals, self.pose points to self.pose_inter. self.pose_inter can be modified as we want.
        
        Arguments
        file_name: If you want to save to a specific file name, set this argument. Otherwise, the self.ses object will be used to determine the file name.
        verbose: print a lot of information
        pose_file_extension: the extension of the file you want to load. If not set, self.pose_file_extension is used. If set, it will modify the value of self.pose_file_extension and load the appropriate pose file. If you give file_name, setting pose_file_extension will have no effect.
        """
        if file_name is None and self.ses is None:
            raise ValueError("self.ses is not set and no file name is given")
        
        if file_name is not None: # use the file_name provided
            fn = file_name
        else: # generate the file name from the information we have
            if pose_file_extension is not None:
                self.pose_file_extension = pose_file_extension
                
            fn = self.ses.fileBase+self.pose_file_extension
        
        
        if not os.path.exists(fn):
            raise OSError(fn+" is missing")
        
        if verbose:
            print("Loading original pose from",fn)
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
        self.occupancy_bins = [np.arange(xy_min[0],xy_max[0]+cm_per_bin,cm_per_bin), # we add cm_per_bin so that it extend to the max and does not cut before
                               np.arange(xy_min[1],xy_max[1]+cm_per_bin,cm_per_bin)]
        
        
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
     
    
    
    
    def occupancy_histogram_1d(self, cm_per_bin =2, smoothing_sigma_cm = 2, smoothing = True, zero_to_nan = True,
                        x_range=None,linspace=False,n_bins=None):
        """
        Function to calculate an occupancy histogram for x position data. 
        The occupancy histogram is a 1D array covering the entire environment explored by the animal.
        Each bin of the array contains the time in seconds that the animal spent in the bin.
        The occupancy histogram is used to calculate firing rate histogram (1d)
               
        
        Arguments
        cm_per_bin: cm per bins in the occupancy map
        smoothing_sigma_cm: standard deviation of the gaussian kernel used to smooth the occupancy map
        smoothing: boolean indicating whether or not smoothing should be applied to the occupancy map
        zero_to_nan: boolean indicating if occupancy bins with a time of zero should be set to np.nan
        x_range: 1D np.array of size 2 [xmin,xmax] with the minimal and maximal x values that should be in the occupancy histogram, default is None and the values are calculated from the data.         
        linspace: whether to use np.linespace instead of np.arange to get the bins. I had to introduce this as I was getting inconsistent number of bins in the maps when the x_range values were decimals. If linspace is True, cm_per_bin is not used and n_bins is used instead.
        n_bins: if using np.linspace, this will be the number of bins in the histogram. If linspace is False, n_bins is not used, cm_per_bin is used instead
        
        Return
        self.occupancy_histo is set. It is a 1D numpy array containing the time spent in seconds in a set of bins covering the environment
        """
        
        if self.pose is None:
            raise TypeError("Set the self.pose array before attempting to calculate the occupancy_histogram_1d")
        
        # we save this for later use when calculating firing rate maps
        self.occupancy_cm_per_bin=cm_per_bin
        self.occupancy_smoothing = smoothing
        self.smoothing_sigma_cm = smoothing_sigma_cm
        
        # remove invalid position data
        invalid = np.isnan(self.pose[:,1])
        #print("{} invalid rows out of {}, % invalid: {:.2f}".format(invalid.sum(),invalid.shape[0],invalid.sum()/invalid.shape[0]*100 ))
        val = self.pose[~invalid,1]
        
        ## determine the size of the occupancy map with the minimum and maximum x and y values
        #print("max x and y values: {}".format(val.max(axis=0)))
        if x_range is None:
            x_max = np.ceil(val.max())+cm_per_bin
            x_min = np.floor(val.min())-cm_per_bin
        else :
            x_max= x_range[1]
            x_min= x_range[0]
        
        # create two arrays that will be our bin edges in the histogram function
        if linspace:
            self.occupancy_bins = np.linspace(x_min,x_max,n_bins+1)
        else :
            self.occupancy_bins = np.arange(x_min,x_max+cm_per_bin,cm_per_bin)               
        
        # calculate the occupancy map
        occ,x_edges = np.histogram(val,bins= self.occupancy_bins)
        
        # calculate the time per sample
        sec_per_sample = self.pose[1,0]-self.pose[0,0] # all rows have equal time intervals between them, we get the first one
        #print("time per sample: {}".format(sec_per_sample))
        
        # get the time in seconds
        occ = occ*sec_per_sample
    
        # smoothin of occupancy map
        if smoothing:
            occ_sm = ndimage.gaussian_filter1d(occ,sigma=smoothing_sigma_cm/cm_per_bin,mode="nearest")
                     
        else:
            occ_sm = occ # if no smoothing just get a reference to occ, because we want to use occ_sm for the rest of the function
          
        # set bins at 0 to np.nan
        if zero_to_nan:
            occ_sm[occ==0] = np.nan
    
        # save the occupancy map for later use
        self.occupancy_histo = occ_sm
    
    
    
        
    def occupancy(self, environment_shape=None):
        """
        Function to calculate the proportions of bins of the occupancy map covered by the animal. Can be used for rectanglular and circular arenas.
        
        Arguments
        arena: specifies the shape of the arena:circle/rectangle /square      
        
        Return
        occupancy
        """
        if not hasattr(self, 'occupancy_map'):
            raise TypeError('You have to call ap.occupancy_map_2d() before calling this function')
        
        if environment_shape == 'rectangle' or environment_shape=='square':
            area = self.occupancy_map.shape[0]*self.occupancy_map.shape[1] # area of a rectangle

        elif environment_shape == 'circle':
            # use the smaller dimension as diameter of the circle as there might be reflections outside the arena
            area = ((np.min(self.occupancy_map.shape)/2)**2)*np.pi # area of a circle
            
        else:
            raise TypeError("This arena shape is not supported. Only square, rectangle or circle can be used.")

        occupancy = self.occupancy_map[~np.isnan(self.occupancy_map)].shape[0]/area
        
        return occupancy
    
    
    
    def head_direction_occupancy_histogram_per_occupancy_bin(self):
        """
        Function to calculate the hd occupancy histogram for each bin in the x-y occupancy map.
        
        No arguments       
        
        Return
        Nothing. The ap.hd_occupancy_histogram_per_occupancy_bin 3D array is set.
        """
        if not hasattr(self, 'occupancy_map'):
            raise TypeError('You have to call ap.occupancy_map_2d() before calling this function')
        if not hasattr(self, 'hd_occupancy_histogram'):
            raise TypeError('You have to call ap.head_direction_occupancy_histogram() before calling this function')
        
        deg_per_bin = self.hd_occupancy_deg_per_bin
        # create an empty 3d array to fill with the data
        array=np.zeros(self.occupancy_map.shape[0]*self.occupancy_map.shape[1]*int(360/deg_per_bin))
        pose_hd_hist=np.reshape(array, (self.occupancy_map.shape[0],self.occupancy_map.shape[1],int(360/deg_per_bin)))

        #include position data only if hd value is valid
        invalid = np.isnan(self.pose[:,4])
        pose_val=self.pose[~invalid,1:3]
        hd_val=self.pose[~invalid,4]

        #calculate the hd occupancy histogram for each bin in the occupancy map 
        for i,j in np.nditer(np.meshgrid(range(self.occupancy_map.shape[0]), range(self.occupancy_map.shape[1]))):
            val=[]
            if not np.isnan(self.occupancy_map[i][j]):
                val = hd_val[(pose_val[:,0]>=self.occupancy_bins[0][i]) & (pose_val[:,0]<self.occupancy_bins[0][i+1]) & (pose_val[:,1]>=self.occupancy_bins[1][j]) & (pose_val[:,1]<self.occupancy_bins[1][j+1])]

            occ,edges = np.histogram(val, bins=self.hd_occupancy_bins)
            pose_hd_hist[i,j,:]=occ
        
        self.hd_occupancy_histogram_per_occupancy_bin = pose_hd_hist
     
    def load_ttl_ups_files(self, ses = None, add_trial_offset=True):
        """
        Function to load ttl_ups data from data saved to file
        
        The ttl_ups files are created in self.pose_from_positrack_files() by scanning the synch channel of the .dat files for ttl pulses.
        
        The values in ttl_ups are the time in samples within each .dat file.
        
        Arguments:
        ses: a spikeA.session
        add_trial_offset: whether to add the trial offset to the ttl ups. 
        """
         
        if ses is None and self.ses is None:
            raise TypeError("Please provide a session object with the ses argument")
        
        if ses is not None:
            if not (issubclass(type(ses),Session) or isinstance(ses,Session)): 
                raise TypeError("ses should be a subclass of the Session class")
        
        self.ttl_ups = []
        trial_sample_offset = 0
        
        for i,t in enumerate(self.ses.trial_names):
            up_file_name = self.ses.path + "/" + t+".ttl_up.npy"
            dat_file_name = self.ses.path + "/" + t+".dat"
            df = Dat_file_reader(file_names=[dat_file_name],n_channels = self.ses.n_channels)
            ttls = np.load(up_file_name)
            if add_trial_offset:
                ttls = ttls+trial_sample_offset
            
            self.ttl_ups.append(ttls)
            trial_sample_offset+=df.total_samples
        

    def pose_from_positrack_files(self,ses=None, ttl_pulse_channel=None, interpolation_frequency_hz = 50, extension= "positrack2", use_previous_up_detection=True,transform_to_cm=True):

        """
        Method to calculute pose at fixed interval from a positrack file.
        
        Arguments
        ses: A Session object
        ttl_pulse_channel: channel on which the ttl pulses were recorded. If not provided, the last channel is assumed
        interpolation_frequency_hz: frequency at which with do the interpolation of the animal position
        extension: file extension of the file with position data (positrack or positrack2)
        use_previous_up_detection: if True, it will look for a file containing the time of ttl pulses instead of detecting the ttl pulses from the dat file (slow)
        transform_to_cm: if True, the data will be assumed to enter as pixels and will be transformed into cm based on px_per_cm array or float as by the config.
                
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
        # list to store positrack times
        self.pt_times = []
        
        # interpolate to have data for the entire .dat file (padded with np.nan at the beginning and end when there was no tracking)
        interpolation_step = self.ses.sampling_rate/interpolation_frequency_hz # interpolate at x Hz
        print("Interpolation step: {} samples".format(interpolation_step))
        
        print("")
        print("Loop through {} trials".format(self.ses.n_trials))
        print("")


        #loop for trials
        for i,t in enumerate(self.ses.trial_names):
            dat_file_name = self.ses.path + "/" + t+".dat"
            positrack_file_name = self.ses.path + "/" + t+"."+ extension
            print(dat_file_name)
            print(positrack_file_name)

            positrack_file = Path(positrack_file_name)
            
            if not positrack_file.exists() :
                raise OSError("positrack file {} missing".format(positrack_file_name))

            
            # get ttl pulses from dat file or from previously stored file (much faster)
            df = Dat_file_reader(file_names=[dat_file_name],n_channels = self.ses.n_channels)
            up_file_name = self.ses.path + "/" + t+".ttl_up.npy"
            up_file = Path(up_file_name)
            
            if use_previous_up_detection and up_file.exists():
                # read up file
                print("Getting ttl pulses time from", up_file)
                ttl = np.load(up_file)
            else:
                # get the ttl pulses from .dat file (on ttl_pulse_channel or last channel per default)
                if ttl_pulse_channel is None:
                    ttl_pulse_channel = self.ses.n_channels-1
                ttl_channel_data = df.get_data_one_block(0,df.files_last_sample[-1],np.array([ttl_pulse_channel]))
                ttl,downs = detectTTL(ttl_data = ttl_channel_data)
            if up_file.exists() == False:
                # save up file for future reading
                print("Saving",up_file)
                np.save(up_file, ttl)                    

            print("Number of ttl pulses detected: {}".format(ttl.shape[0]))
                
            # read the positrack file
            if extension=="positrack":
                pt = pd.read_csv(positrack_file_name, delimiter=" ", index_col=False)

            elif extension=="positrack2" or extension=="positrack2_post" or extension=="positrack_kf" or extension=="positrack2_kf":
                pt = pd.read_csv(positrack_file_name)
                
            elif extension=="trk":
                data = np.reshape(np.fromfile(file=positrack_file_name,dtype=np.int32),(-1,21))
                data = data.astype(np.float32)
                pt = pd.DataFrame({"x":data[:,11], "y":data[:,12],"hd": data[:,10]})
            
            else:
                raise ValueError("extension not supported")
  
            
            # check the positrack file
            print("Number of lines in positrack file: {}".format(len(pt)))
    
            # check that positrack is within recording (start recording, *, start positrack, ..., stop positrack, *, stop recording); *=min 100ms
            startPosi = ttl[0]/self.ses.sampling_rate
            print("start tracking at: {:.4f} sec".format(startPosi))
            timeToEnd = (df.total_samples-ttl[-1])/self.ses.sampling_rate
            print("last ttl to end of dat file duration: {:.4f} sec".format(timeToEnd))
            if startPosi < 0.1:
                print("positrack process was started too early, maybe before start of ktan recording .dat file")
            if timeToEnd < 0.1:
                print("positrack process did not stop before the end of .dat file")

            # check if the number of ttl pulses matches number of video frames
            if len(ttl) != len(pt):
                problem = True
                print("!!!\nalignment problem\n!!!")
                
                delta = len(pt) - len(ttl)
                if delta > 0:
                    print("{} more video frames than ttl pulses".format(delta))
                else:
                    print("{} more ttl pulses than video frames".format(-delta))
                
                print("first ttl sample: {}".format(ttl[0]))
                print("last ttl sample: {}".format(ttl[-1]))
                print("samples in dat file: {}".format(df.total_samples))
                            
                
                
                # if there are just some ttl pulses missing from positrack, copy the last lines in the positrack file
                
                ########################################################
                # We can't fix synchronization problem this way        #
                # by simply adding random line at the end              #
                # how much can the head-direction can change in 666 ms #
                # I would say we add a max of 5 lines                  #
                ########################################################
                
                # allow for a maximum 5 difference of ttl and positrack lines in either way
                if (extension =="positrack" or extension=="positrack2" or extension=="positrack2_kf" or extension=="positrack2_post" or extension=="positrack_post" or extension=="positrack_kf"):
                    if (len(pt) < len(ttl) <= len(pt)+5):
                        #~ missing = len(ttl)-len(pt)
                        #~ print("Missing lines:", missing)
                        #~ pt_mod = pt.append(pt[(len(pt)-missing):])
                        #~ print("Number of lines in adjusted positrack file:", len(pt_mod))
                        #~ pt = pt_mod
                        #~ print("Alignment problem solved by adding "+str(missing)+" ttl pulses to positrack")
                        ttl = ttl[:len(pt)]
                        print("Alignment problem solved by deleting superfluent ttl pulses")
                        problem = False
                    elif (len(ttl) < len(pt) <= len(ttl)+5):
                        # more positrack frames than ttl pulses
                        pt_mod = pt[:len(ttl)] # just take as many frames as needed
                        print("Number of lines in adjusted positrack file:", len(pt_mod))
                        pt = pt_mod
                        print("Alignment problem solved by deleting superfluent frames in positrack")
                        problem = False
                        
                    # do NOT touch original files
                    #~ original_positrack_file = self.ses.path + "/" + t+"o."+ extension
                    #~ os.rename(positrack_file_name, original_positrack_file)
                    #~ if (extension=='positrack'):
                    #~     pt.to_csv(positrack_file_name, sep=' ')
                    #~ else:
                    #~     pt.to_csv(positrack_file_name, sep=',')
                    

                # we will need more code to solve simple problems 
                #
                #
                if problem:
                    raise ValueError("Synchronization problem (positrack {} and ttl pulses {}) for trial {}".format(pt.shape[0],ttl.shape[0],t))

                    
            
            """
            # positrack time , normally this is not taken into account since the time is synced via ttl, use this to check if no pulses were discarded
            # instead of adding artificial ttl pulses / positrack frames to the positrack file at the end, add them at the correct time, instead
            pt_time=pt["startProcTime"].to_numpy()/1000.
            print("startProcTime (positrack time)")
            print(pt_time)
            print(pt_time.shape)
            
            ttl_time = ttl/self.ses.sampling_rate
            print("ttlpulse/sampling_rate (ttl time)")
            print(ttl_time)
            print(ttl_time.shape)
            
            delta=ttl_time-pt_time
            deltadiff=np.diff(delta)
            print("deltadiff")
            print(deltadiff)
            
            print("max=", np.max(deltadiff),", min=",np.min(deltadiff))                
            """
            
            # create a numpy array with the position data (in the dataframe, the columns "x", "y", "hd" always exists and are needed)
            #~ d = np.stack([pt["x"].values,pt["y"].values,pt["hd"].values]).T
            d = np.array(pt[['x', 'y', 'hd']])
            
            if extension=="positrack":
                if 'frame_no' in pt.columns:
                    # This is actually a positrack2 file that was renamed. Do nothing
                    pass
                else:
                    # This is a positrack file, from the original positrack program.
                    # set invalid values to np.nan (Note that invalid data points are set to -1.0 in positrack files - does not apply for positrack2)
                    d[d==-1.0] = np.nan
                    # To keep movement heading consistent with head-direction, we need to reverse the head-direction
                    #~ pt["hd"] = -pt["hd"]
                    d[:,2] = -d[:,2]

            # get the positrack acquisition time (not needed for ktan/ttl- dat syncing, but for other data that is in rostime ( = nanoseconds since epoch ))
            if extension=="positrack2":
                #print("extension is positrack2")
                positrack_time = np.array(pt["acq_time_source_0"]) # might be one of: acq_time_source_0, acq_time_source_1, acq_time_source_2, processing_start_time
                #print("positrack_time",positrack_time.shape)
                print("get positrack time from",positrack_time[0],"to",positrack_time[-1]," (duration = ",positrack_time[-1]-positrack_time[0],")")
                self.pt_times.extend(positrack_time)

            # if data are in degrees, turn into radians (this should actually be always the case: HD data in positrack and positrack2 are stored in degrees)
            hdMin, hdMax = np.nanmin(d[:,2]),np.nanmax(d[:,2])
            hdRange = hdMax - hdMin
            print("hdRange: {}".format(hdRange))
            if hdRange > 2*np.pi + 0.1: # to avoid numerical wrong comparision, range is either 2pi or 360
                print("degree to radian transformation")
                d[:,2] = d[:,2]/180*np.pi
                
            # we want to treat hd values as cos and sin for interpolation
            c = np.cos(d[:,2]) # x component of angle
            s = np.sin(d[:,2]) # y component of angle
            # add cos and sin to our d array
            x = np.stack([c,s]).T
            d = np.hstack([d,x]) # now x, y , hd, cos(hd), sin(hd)
            
            # check for valid and invalid values (valid line = all values are finite so not nan or inf in all columns (x,y,hd))
            valid = np.sum(np.all(np.isfinite(d),axis=1)) #~ valid = np.sum(~np.isnan(d))
            invalid = len(d)-valid #~ invalid = np.sum(np.isnan(d))
            prop_invalid = invalid/len(d) # 1-valid/len(d) #~ prop_invalid = invalid/d.size
            print("Invalid values: {}".format(invalid))
            print("Valid values: {}".format(valid))
            print("Percentage of invalid values: {:.3}%".format(prop_invalid*100))
            if prop_invalid > 0.05:
                print("****************************************************************************************")
                print("WARNING")
                print("The percentage of invalid values is very high. The quality of your data is compromised.")
                print("Solve this problem before continuing your experiments.")
                print("****************************************************************************************")  
                
            # either apply an individual px_per_cm or use one for all trials
            if transform_to_cm == True:
                if isinstance(self.ses.px_per_cm, np.ndarray):
                    px_per_cm = self.ses.px_per_cm[i]
                else:
                    px_per_cm = self.ses.px_per_cm
                print("transforming pixels to cm with px_per_cm:",px_per_cm)
                d[:,[0,1]] /= px_per_cm # transform to cm (for this trial)

            # estimate functions to interpolate
            fx = interp1d(ttl[:], d[:,0], bounds_error=False) # x we will start at 0 until the end of the file
            fy = interp1d(ttl[:], d[:,1], bounds_error=False) # y
            fhdc = interp1d(ttl[:], d[:,3], bounds_error=False) # cos(hd)
            fhds = interp1d(ttl[:], d[:,4], bounds_error=False) # sin(hd)

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
            
            
            print("")
        

        # put all the trials together
        posi = np.concatenate(posi_list)
        print("shape of position data for all trials:",posi.shape)
        
        # sync positrack time
        ttl_all = np.concatenate(self.ttl_ups) # append->extend (merge lists)
        if (len(self.pt_times) == len(ttl_all)): # if pt_times was filled
            ptime2time = interp1d(self.pt_times, ttl_all, bounds_error=False) # transform positrack time to time used here (ktan dat time, 0=start of dat recording)
            # load some external timing in the positrack reference frame (seconds/nanoseconds since epoch)
            logfile=self.ses.path + "/times.log"
            if os.path.exists(logfile):
                print("use logfile times from",logfile)
                times=np.loadtxt(logfile)
                print("got list of times, len =",len(times))
                # and convert it to our time frame (feed it to the time conversion by pt->ttl)
                times_ = ptime2time(times) / self.ses.sampling_rate
                self.ses.log_times = times_ # save logged times to pose
                print("converted times (from",times_[0]," to",times_[-1],") , shape:", times_.shape)
                times_fn = self.ses.path + "/times.npy"
                np.save(times_fn, times_)
                print("saved to",times_fn)

        # if we have more than 1 trial, we need to re-interpolate so that we constant time difference between position data
        #if ses.n_trials > 1:

        # estimate functions to interpolate
        fx = interp1d(posi[:,0], posi[:,1], bounds_error=False) # x we will start at 0 until the end of the file
        fy = interp1d(posi[:,0], posi[:,2], bounds_error=False) # y 
        fhdc = interp1d(posi[:,0], posi[:,3], bounds_error=False) # cos(hd)
        fhds = interp1d(posi[:,0], posi[:,4], bounds_error=False) # sin(hd)
        # (might be useful to make these functions available globally, to get the pose at any time, especially for spike times)- pose at spike time (like in tuning curves)

        # new time to interpolate
        nt = np.arange(0, posi[-1,0]+interpolation_step,interpolation_step)

        #
        new_x = fx(nt)
        new_y = fy(nt)
        new_hdc = fhdc(nt)
        new_hds = fhds(nt)

        # get back the angle from the cosin and sin
        new_hd = np.arctan2(new_hds,new_hdc) # np.arctan2(y_value,x_value)
        
        # define function to get pose at any time (useful for other methods in this class)
        #~ self.fx = fx
        #~ self.fy = fy
        #~ self.fhdc = fhdc
        #~ self.fhds = fhds
        
        
        # contain time, x,y,z, yaw, pitch, roll
        # index:    0   1,2,3, 4,    5,     6
        self.pose_ori = np.empty((new_x.shape[0],7),float) # create the memory
        self.pose = self.pose_ori # self.pose points to the same memory as self.pose_ori
        
                
        ##the hd data should be aligned with the position data
        ##in positrack this is currently not the case, as you can see via this code:
        #distance = np.diff(ap.pose[:,1:3], axis=0, append=np.nan)
        #movement_direction= np.arctan2(distance[:,1],distance[:,0])
        #hd = ap.pose[:,4]
        #plt.scatter(movement_direction,hd, s=1)
        ##if the hd data and the position data were aligned, you could fit a linear function going through the origin
        ##to achieve this, the hd data need to be shifted by -pi/2
        #~ if extension=="positrack":
        #~     new_hd=new_hd-np.pi/2
        #~     new_hd[new_hd<=-np.pi]=new_hd[new_hd<=-np.pi]+2*np.pi
        #if extension=="positrack":
        #    new_hd = -new_hd
        
        self.pose[:] = np.nan
        self.pose[:,0] = nt/self.ses.sampling_rate # from sample number to time in seconds
        self.pose[:,1] = new_x # /self.ses.px_per_cm # transform to cm (this is done above for each trial)
        self.pose[:,2] = new_y # /self.ses.px_per_cm # transform to cm (this is done above for each trial)
        self.pose[:,4] = new_hd

        
        ## create intervals that cover the entire session
        if self.intervals is not None:
            # set default time intervals from 0 to the last sample
            self.set_intervals(inter=np.array([[0,self.pose[:,0].max()+1]]))
        else :
             # get intervals for the first time
            self.intervals = Intervals(inter=np.array([[0,self.pose[:,0].max()+1]]))
            
    
    def pose_at_time(self, t_sec):
        
        """
        Method to get the pose at arbitrary time
        Must be called after interpolate_pose
        
        Arguments: t_sec (time in seconds) might be numpy array of several n time points
        Returns: (3,n) or (3) numpy array with x,y,hd as rows
        """
        
        if not hasattr(self, 'fx'):
            raise TypeError("You need to call ap.interpolate_pose before calling ap.pose_at_time(t)")
        
        # t_sec : time in seconds
        # t : time in samples = t_sec * sampling_rate
        # t = t_sec * self.ses.sampling_rate
        t=t_sec
        
        x = self.fx(t)
        y = self.fy(t)
        hd = self.fhd(t)
        
        return np.array([x,y,hd]).squeeze()
    
        """
        # code example to verify
        ap.interpolate_pose()
        t=np.linspace(100,2000,5000)
        x,y,hd = ap.pose_at_time(t)
        #plt.plot(t,x)
        #plt.plot(t,y)
        plt.plot(t,hd)
        plt.plot(ap.pose[:,0],ap.pose[:,4])
        plt.xlim(200,250)
        """
    
    def interpolate_pose(self):
        
        """
        Method to make the pose available at arbitrary time points by interpolation.
        Should be called after load_pose_from_file() or pose_from_positrack_files()
        Call then using pose_at_time
        
        Arguments: None
        Returns: nothing, but sets self.fx, self.fy, self.fhd
        """
        
        # For the class Spatial_properties (mostly using "n.spatial_properties" when n is a Neuron instance), these methods set these attributes
        # spike_head_direction() - spike_hd (1D array)
        # spike_position() - spike_posi (2D array)
        
        if self.pose is None:
            raise TypeError("Set the self.pose array before attempting to interpolate pose")

        self.fx = interp1d(self.pose[:,0], self.pose[:,1], bounds_error=False) # x
        self.fy = interp1d(self.pose[:,0], self.pose[:,2], bounds_error=False) # y 
        fhdc    = interp1d(self.pose[:,0], np.cos(self.pose[:,4]), bounds_error=False) # cos(hd)
        fhds    = interp1d(self.pose[:,0], np.sin(self.pose[:,4]), bounds_error=False) # sin(hd)
        self.fhd = lambda t: np.arctan2(fhds(t),fhdc(t))
        
        
        
    def filter_pose(self, windowlen_sec = .25, filter_hd=True):
        """
        apply median filter on x,y pose (useful for filtering outliers)
        windowlen_sec : median window in seconds (will be transformed to discrete time steps using dt)
        filter_hd: also filter HD
        """
        time = self.pose[:,0]
        dtime = np.diff(time)[0]
        xvals,yvals,hd = self.pose[:,1],self.pose[:,2],self.pose[:,4]
        windowlen_ind = (int(windowlen_sec/dtime) // 2) * 2  + 1 # odd number
        xvals_ = medfilt(xvals, windowlen_ind)
        yvals_ = medfilt(yvals, windowlen_ind)
        self.pose[:,1] = xvals_
        self.pose[:,2] = yvals_
        
        hd_cos = np.cos(hd)
        hd_sin = np.sin(hd)
        hd_cos_ = medfilt(hd_cos, windowlen_ind)
        hd_sin_ = medfilt(hd_sin, windowlen_ind)
        hd_ = np.arctan2(hd_sin_, hd_cos_)
        if filter_hd:
            self.pose[:,4] = hd_
        
    def hd_use_speedvector(self):
        """
        use movement direction as HD (experimental)
        """
        distance = np.diff(self.pose[:,1:3], axis=0, append=np.nan)
        movement_direction = np.arctan2(distance[:,1],distance[:,0])
        self.pose[:,4] = movement_direction

    def correct_hd_flip(self, min_speed = 7, max_speed = 100, plot=False):
        """
        correct HD if it is flipping (deviates more than 90° from movement direction), make sure it is always closer to movement direction than the opposite direction (180°)

        min_speed, max_speed: only apply to these values
        plot: show plot

        Returns:
        number of indices flipped
        """

        t= self.pose[:,0]
        x= self.pose[:,1]
        y= self.pose[:,2]
        hd= self.pose[:,4]

        xd = np.diff(x,append=np.nan)
        yd = np.diff(y,append=np.nan)
        td = np.diff(t,append=np.nan)
        heading = np.arctan2(yd,xd)
        speed= np.sqrt(xd**2+yd**2)/td

        indices = (speed <= max_speed) & (speed >= min_speed)
        #~ sum(indices), len(indices)

        # signed angle difference
        delta = range_pi(hd-heading)

        # delta more than 90° (=pi/2) means HD is closer to opposite movement direction, so flip HD to match movement direction better.
        hd_ = hd.copy()
        #~ indices_correct = (np.abs(delta) > np.pi/2)
        indices_correct = (np.abs(delta) > np.pi/2) & indices # these indices should be corrected
        hd_[indices_correct] -= np.pi # flip
        hd_ = range_pi(hd_)
        
        delta_ = range_pi(hd_-heading) # new delta (now absolute value <= pi/2)

        medDelta = np.nanmedian(np.abs(delta))
        medDelta_ = np.nanmedian(np.abs(delta_))

        if plot:
            fig, axes = plt.subplots(nrows=1, ncols=4,figsize=(12,3), constrained_layout=True)

            axes[0].scatter(heading[indices],hd[indices],s=1,alpha=0.1)
            #~ axes[0].scatter(heading[~indices],hd[~indices],s=.5,alpha=0.05,c='grey')
            axes[0].set_xlabel("Movement heading")
            axes[0].set_ylabel("Head direction")
            axes[0].set_title("before")

            axes[1].hist(np.abs(delta),bins=30)
            axes[1].axvline(x=medDelta, c='blue')
            axes[1].set_xlabel("Delta HD-heading")
            axes[1].set_title("Median delta {:.3f} = {:.1f}°".format(medDelta,np.rad2deg(medDelta)))

            axes[2].scatter(heading[indices],hd_[indices],s=1,alpha=0.1)
            axes[2].set_xlabel("Movement heading")
            axes[2].set_ylabel("Head direction")
            axes[2].set_title("after")

            axes[3].hist(np.abs(delta_),bins=30)
            axes[3].axvline(x=medDelta_, c='blue')
            axes[3].set_xlabel("Delta HD-heading")
            axes[3].set_title("Median delta {:.3f} = {:.1f}°".format(medDelta_,np.rad2deg(medDelta_)))

            plt.show()

        self.pose[:,4] = hd_ # update
        return indices_correct
        
            
    def speed_from_pose(self, sigma=1):
        """
        Method to calculute the speed (in cm/s) of the animal from the position data
        The speed at index x is calculated from the distance between position x and x+1 and the sampling rate.
        Then a Gaussian kernel is applied for smoothing.
        
        Arguments
        sigma: for Gaussian kernel
                
        Return
        No value is returned but self.speed and self.distance is set
        
        """
        if self.pose is None:
            raise TypeError("Set the self.pose array before attempting to calculate the speed")
        
        # create empty speed array
        self.speed = np.empty((len(self.pose[:,0]),1),float)
        
        
        # calculate the time per sample
        sec_per_sample = self.pose[1,0]-self.pose[0,0] # all rows have equal time intervals between them, we get the first one
        
        # calculate the distance covered between the position data and divide by time per sample
        delta = np.diff(self.pose, axis=0, append=np.nan)
        distance = np.sqrt(delta[:,1]**2 + delta[:,2]**2)
        # speed = distance/sec_per_sample
        
        # apply gaussian filter for smoothing
        distance = gaussian_filter1d(distance, sigma=sigma)
        self.speed = distance / sec_per_sample
        self.distance = np.nancumsum(distance) # total distance
    
    
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

    
    def invalid_outside_spatial_area(self, environment_shape=None, radius=None, length=None, center=None):
        """
        Method that set the position data (self.pose[:,1:7]) outside a defined zone to np.nan.
        
        The area can be a circle or a rectangle.
        
        To undo, call self.unset_intervals() or self.set_intervals(). unset_intervals() and set_intervals() use the data stored in self.ori_pose
        
        This function should be called **after** setting any relevant Intervals for the analysis.
        
        Arguments:
        environment_shape: "circle","rectangle","square"
        radius: radius of a circle, only needed if working with circle
        length: tuple with the length in x and y direction
        center: 1D np.array of size 2, [x,y], center of a circle/rectangle
        
        Return:
        Nothing is returned. self.pose[,1:7] are set to np.nan if the animal is not in the zone.
        """
        valid_shapes = ["circle","square","rectangle"]
        
        if not environment_shape in valid_shapes:
            raise ValueError("environment_shape should be part of the list {}".format(valid_shapes))
        
        # if center is not specified, determine from center of mass of the occupancy map
        if center is None:
            self.occupancy_map_2d(cm_per_bin=1)
            occ_map = self.occupancy_map.copy()
            # nan values need to be 0
            occ_map[np.isnan(occ_map)]=0.0
            # the occupancy map is rotated by -90° to the pose date. We need to rotate back.
            occ_map=ndimage.rotate(occ_map, 90)
            # get the center of mass of the occupancy map
            center_occ=ndimage.center_of_mass(occ_map)
            #print(center_occ)
            # we need to transform back from the occupancy map to the pose data. 
            # The y axis starts in the top left orner for the occ_map, but in the bottom left corner for the pose data.
            x_range_pose = np.nanmax(self.pose[:,1]-np.nanmin(self.pose[:,1]))
            x_range_occ = occ_map.shape[1]
            y_range_pose = np.nanmax(self.pose[:,2]-np.nanmin(self.pose[:,2]))
            y_range_occ = occ_map.shape[0]
            center = (np.nanmin(self.pose[:,1])+center_occ[0]*x_range_pose/x_range_occ,np.nanmax(self.pose[:,2])-center_occ[1]*y_range_pose/y_range_occ)
            #print(center)

                
        # deal with circle
        if environment_shape == "circle":
            if radius is None:
                raise ValueError("set the radius argument")
            
            # calculate distance to center
            dist = np.sqrt((self.pose[:,1]-center[0])**2 + (self.pose[:,2]-center[1])**2)
            # outside circle = np.nan
            self.pose[dist>radius,1:7] = np.nan
            
            r = radius
            
        # deal with rectangle
        if environment_shape == "rectangle" or environment_shape=="square":
            if length is None:
                raise ValueError("set the length argument")
            if isinstance(length,int) or isinstance(length,float):
                length=(length,length)
            
            # set pixels outside rectangle of length length np.nan
            self.pose[self.pose[:,1] > center[0]+length[0]/2, 1:7] = np.nan
            self.pose[self.pose[:,2] > center[1]+length[1]/2, 1:7] = np.nan
            self.pose[self.pose[:,1] < center[0]-length[0]/2, 1:7] = np.nan
            self.pose[self.pose[:,2] < center[1]-length[1]/2, 1:7] = np.nan
            
            r=np.asarray([length[0]/2,length[1]/2])


        center = np.array(center) # tuple to numpy array
        if environment_shape=="circle":
            xy_range = np.array([center - r, center + r]) # square that covers the valid range
        if environment_shape=='rectangle' or environment_shape=='square':
            xy_range = np.array([center - r, center + r]) # rectangle that covers the valid range
        return xy_range # useful to have this for restricting the area later


    
    def pose_inside_spatial_area(self, environment_shape=None, radius=None, length=None, center=None):
        """
        similar as invalid_outside_spatial_area, but leaves pose unchanged (does not set pose to nan)
        
        Returns: indices of pose where spatial condition is satisfied
        """

        valid_shapes = ["circle","square","rectangle"]

        if not environment_shape in valid_shapes:
            raise ValueError("environment_shape should be part of the list {}".format(valid_shapes))

        if environment_shape == "circle":
            if radius is None:
                raise ValueError("set the radius argument")

            # calculate distance to center
            dist = np.sqrt((self.pose[:,1]-center[0])**2 + (self.pose[:,2]-center[1])**2)
            
            # return indices where distance is not greater than radius (dist <= radius)
            return ~(dist>radius)

        # deal with rectangle
        if environment_shape == "rectangle" or environment_shape=="square":
            if length is None:
                raise ValueError("set the length argument")
            if isinstance(length,int) or isinstance(length,float):
                length=(length,length)

            # return indicies where pose is in rectangle of length length (not one of the 4 conditions)
            return ~(self.pose[:,1] > center[0]+length[0]/2 | self.pose[:,2] > center[1]+length[1]/2 | self.pose[:,1] < center[0]-length[0]/2 | self.pose[:,2] < center[1]-length[1]/2)


    
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
        #~ loc_vector = np.array([np.cos(loc)],np.sin(loc))
        #~ hd = elf.pose[:,4] # np.deg2rad?
        #~ hd_vector = np.array([np.cos(hd),np.sin(hd)])        
        #~ delta = np.arccos(hd_vector@loc_vector)
        #~ self.pose[delta>sigma,1:7] = np.nan        
        
        diffxy = np.diff(self.pose[:,1:3], axis=0).transpose() # np.arctan2(dx,dy)
        hd = np.deg2rad(self.pose[:,4])
        hd = hd-np.pi
        hd_vector = np.array([np.cos(hd),np.sin(hd)])

        delta_angles_hd_move = np.arccos(np.array([np.dot(h,d) for h,d in zip(hd_vector[:,:-1].T,diffxy.T)]))
        # plt.hist(delta_angles_hd_move)
        delta = np.abs(delta_angles_hd_move-np.pi/2)
        # plt.hist(delta)
        
        # sigma = np.pi/4
        # np.sum(delta > sigma)
        
        self.pose[np.append(delta>sigma,False),1:7] = np.nan
        
        
    def find_xy_range(self, diameter=None):        
        """
        method to find suitable square in pose
        # find good shape automagically
        what it does:
        1. find mean location of pose (each time counts the same, so it is like the weighted occupancy map with weight= time spent in bin)
        2. create a square with diameter around mean location
        3. if this square is bigger in any direction as the actual data, correct this edge to the appropriate actual pose
         (moves the rect to the closest edge in pose)
        
        Returns xmin,ymin;xmax,ymax (can be used as xy_range to calculate occupancy map and firing rate map - use to crop image)
        """
        
        # find center
        self.meanloc = np.nanmean(self.pose[:,1:3], axis=0)
        xmean,ymean = self.meanloc
        # print("pose mean",xmean,ymean)
        # print("meanloc-rect, xymin,xymax",self.meanloc - diameter/2. , self.meanloc + diameter/2.)
        
        # find the minimum and maximum x and y values in the pose
        val = self.pose[:,1:3]
        to_add = self.occupancy_cm_per_bin if (self.occupancy_cm_per_bin and np.isfinite(self.occupancy_cm_per_bin)) else 0.
        xy_max = np.ceil(np.nanmax(val,axis=0))  - to_add
        xy_min = np.floor(np.nanmin(val,axis=0)) + to_add
        self.poserect = xy_min,xy_max
        # print("pose min/max:",xy_min,xy_max)
        
        
        # if the mean square is bigger than the pose min/max in one direction, adapt to pose min/max
        xy_max_ = np.min([xy_max, self.meanloc + diameter/2.], axis=0)
        # xy_min_ = np.max([xy_min, self.meanloc - diameter/2.], axis=0)
        xy_min_ = np.max([xy_min, xy_max_ - diameter], axis=0) # use corrected meanloc variable as second arg in max to ensure square shape
        xy_max_ = xy_min_ + diameter # correct to square
        
        # print("xy min/max:",xy_min_,xy_max_)

        # make square
        # xymean_ = np.mean([xy_min_, xy_max_], axis=0)
        # return xymean_ - diameter/2., xymean_ + diameter/2.
        
        # return xy_min_, xy_max_
        return np.array([xy_min_, xy_max_])
    
    
    def find_xy_range_2(self, diameter, chunks=1000):
        """
        method to find shape in pose by splitting into chunks and applying median on them
        diameter: diameter or length of box
        
        Returns: center, xy_range of environment
        """
        pose_xy = self.pose[:,[1,2]]
        #~ (xmin,ymin),(xmax,ymax) = np.nanmin(pose_xy, axis=0), np.nanmax(pose_xy, axis=0)
        #~ np.nanmedian(pose_xy, axis=0)

        pose_xy_split = np.array_split(pose_xy, chunks)
        pose_xy_split_medians = np.array([np.nanmedian(p,axis=0) for p in pose_xy_split])
        xy_min_max = np.nanmin(pose_xy_split_medians, axis=0), np.nanmax(pose_xy_split_medians, axis=0) # (xmin,ymin),(xmax,ymax)
        center = np.nanmean(xy_min_max, axis=0)
        xy_range = np.array([center-diameter/2, center+diameter/2])

        return center, xy_range
        
        
    def invalid_ratio(self):
        """
        function to return ratio of invalid values (to be called after setting intervals)
        Returns: the ratio of invalid values
        """
        invalid = np.isnan(self.pose[:,[1,2,4]]).any(axis=1) # invalid values on 1,2,4 = x,y,hd
        return sum(invalid) / len(invalid)

    def mid_point_from_edges(self, edges):
        """
        get histogram midpoints from histogram edges
        """
        #return np.mean(edges, axis=1) # transforms [[1,2],[6,8],[99,100.5]] -> [ 1.5 ,  7.  , 99.75]
        return (edges[1:]+edges[:-1])/2. # transforms array([1, 2, 3, 4, 5, 6, 7, 8, 9]) to array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])

    def coord_cm2bin(self, val_x, val_y):
        # self.occupancy_map.shape
        bins_x, bins_y = self.occupancy_bins
        midpoints_x, midpoints_y = self.mid_point_from_edges(bins_x), self.mid_point_from_edges(bins_y)
        # range_x, range_y = np.transpose([bins_x[:-1], bins_x[1:]]), np.transpose([bins_y[:-1], bins_y[1:]])
        # bin_x, bin_y = np.argmin([ np.abs(val_x - midpoint_x) for midpoint_x in midpoints_x ]), np.argmin([ np.abs(val_y - midpoint_y) for midpoint_y in midpoints_y ])
        ind_x = np.digitize(val_x, bins_x) - 1
        ind_y = np.digitize(val_y, bins_y) - 1
        ret_x = midpoints_x[ind_x] if ind_x in range(len(midpoints_x)) else np.nan
        ret_y = midpoints_y[ind_y] if ind_y in range(len(midpoints_y)) else np.nan
        # return ret_x, ret_y
        return ind_x if ind_x in range(len(midpoints_x)) else np.nan, ind_y if ind_y in range(len(midpoints_y)) else np.nan
    
    def coord_cm2bin_xyvals(self, xy_vals):
        binsx,binsy = np.transpose([self.coord_cm2bin(xy_val[0],xy_val[1]) for xy_val in xy_vals])
        return binsx,binsy
        
    def positrack_type(self,ses=None):
        """
        Function trying to answer if the data were collected with positrack or positrack2 or trk
        
        Return Type of tracking as a string. Can be "positrack", "positrack2", "trk" or "None" 
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
        
        positrack_file_name = self.ses.path + "/" + t+".trk"
        positrack_file = Path(positrack_file_name)
        if positrack_file.exists() :
            return "trk"
        
        return "None"
    
    def times2intervals(self,times):
        """
        transform list of times to all intervals between these points
        Return: corresponding 2d np array
        """
        return np.transpose([times[:-1], times[1:]])

    
    def rotate(self, point, origin, radians):
        """rotate data points"""
        x,y = point; offset_x, offset_y = origin
        adjusted_x = (x - offset_x); adjusted_y = (y - offset_y)
        cos_rad = np.cos(radians); sin_rad = np.sin(radians)
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        return qx, qy
