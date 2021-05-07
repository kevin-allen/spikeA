import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

from spikeA.Dat_file_reader import Dat_file_reader
from spikeA.ttl import detectTTL
from spikeA.Intervals import Intervals
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
        
    Methods:
        pose_from_positrack_file()
        set_intervals()
        unset_intervals()
        
    """
    def __init__(self):
        """
        Constructor of the Animal_pose class
        """
        self.pose = None
        self.pose_ori = None
        self.pose_inter = None
        self.intervals = None
    
    def save_pose_to_file(self):
        pass
    def load_pose_from_file(self):
        pass
    
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
        print("Number of poses: {}".format(self.pose.shape[0]))
    
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
        print("Number of poses: {}".format(self.pose.shape[0]))
        
    
    def occupancy_map_2d(self, cm_per_bin =2, smoothin_std_cm = 1.5):
        """
        Function to calculate an occupancy map for x and y position data.
        
        
        """
    
    
    
    def pose_from_positrack_file(self,ses, ttl_pulse_channel=None, interpolation_frequency_hz = 50):
        """
        Method to calculute pose at fixed interval from a positrack file.
        
        Arguments
        ses: A Session object
        ttl_pulse_channe: channel on which the ttl pulses were recorded. If not provided, the last channel is assumed
        interpolation_frequency_hz: frequency at which with do the interpolation of the animal position
                
        Return
        No value is returned but self.time and self.pose are set
        
        """
        # we loop for each trial, check the syncrhonization, append the position data and ttl time 
        # (taking into account the length of previous files)

        # counter for trial offset from beginning of recordings
        trial_sample_offset = 0
        # list to store the position array of each trial
        posi_list = []

        # interpolate to have data for the entire .dat file (padded with np.nan at the beginning and end when there was no tracking)
        interpolation_step = ses.sampling_rate/interpolation_frequency_hz # interpolate at x Hz
        print("Interpolation step: {} samples".format(interpolation_step))


        # loop for trials
        for i,t in enumerate(ses.trial_names):
            dat_file_name = ses.path + "/" + t+".dat"
            positrack_file_name = ses.path + "/" + t+".positrack"
            print(dat_file_name)
            print(positrack_file_name)

            positrack_file = Path(positrack_file_name)
            if not positrack_file.exists() :
                raise OSError("positrack file {} missing".format(positrack_file_name))

            # read the ttl channel for positrack
            df = Dat_file_reader(file_names=[dat_file_name],n_channels = ses.n_channels)
            ttl_channel_data = df.get_data_one_block(0,df.files_last_sample[-1],np.array([ses.n_channels-1]))
            ttl = detectTTL(ttl_data = ttl_channel_data)
            print("Number of ttl pulses detected: {}".format(ttl.shape[0]))

            # read the positrack file
            pt = pd.read_csv(positrack_file_name, delimiter=" ")
            print("Number of lines in positrack file: {}".format(pt.shape[0]))
            if ttl.shape[0] != pt.shape[0]:
                print("alignment problem")

                # we will need code to solve simple problems 
                #
                #
                raise ValueError("Synchronization problem (positrack and ttl pulse) for trial {}".format(t))

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
            fx = interp1d(ttl[:,0], d[:,0], bounds_error=False) # x we will start at 0 until the end of the file
            fy = interp1d(ttl[:,0], d[:,1], bounds_error=False) # y 
            fhdc = interp1d(ttl[:,0], d[:,3], bounds_error=False) # cos
            fhds = interp1d(ttl[:,0], d[:,4], bounds_error=False) # sin

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

            # change the offset for the next trial
            trial_sample_offset+=trial_sample_offset+df.total_samples

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
        self.pose[:,0] = nt/ses.sampling_rate # from sample number to time in seconds
        self.pose[:,1] = new_x/ses.px_per_cm # transform to cm
        self.pose[:,2] = new_y/ses.px_per_cm # transform to cm
        self.pose[:,4] = new_hd
        
        ## create intervals that cover the entire session
        if self.intervals is not None:
            # set default time intervals from 0 to the last sample
            self.set_intervals(inter=np.array([[0,self.pose[:,0].max()+1]]))
        else :
             # get intervals for the first time
            self.intervals = Intervals(inter=np.array([[0,self.pose[:,0].max()+1]]))