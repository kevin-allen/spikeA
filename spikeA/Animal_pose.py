import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

from spikeA.Dat_file_reader import Dat_file_reader
from spikeA.ttl import detectTTL
class Animal_pose:
    """
    Class containing information about the pose (position and orientation) of an animal in time
    
    The position is in x,y,z and the orientation in Euler angles (yaw,pitch,roll). 
    When not available, data are set to np.NAN
    
    To ease computations, the time intervals between data point is kept constant.
    
    Position data are in cm.
    Angular data are in radians to make computations simpler
    
    Attributes:
    
        time: 1D numpy array with the time stamp of each data point in the pose array
        pose: 2D numpy array, columns are (x,y,z,yaw,pitch,roll)
        inter: Interval object
        
    Methods:
        set_spike_train()
        
    """
    def __init__(self):
        """
        Constructor of the Animal_pose class
        """
        pass
    
    def save_pose_to_file(self):
        pass
    def load_pose_from_file(self):
        pass
    
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

        new_x = fx(nt)
        new_y = fy(nt)
        new_hdc = fhdc(nt)
        new_hds = fhds(nt)

        # get back the angle from the cosin and sin
        new_hd = np.arctan2(new_hdc,new_hds)

        self.time = nt/ses.sampling_rate # from sample number to time in seconds
        self.pose = np.empty((new_x.shape[0],6),float)
        self.pose[:] = np.nan
        self.pose[:,0] = new_x/ses.px_per_cm # transform to cm
        self.pose[:,1] = new_y/ses.px_per_cm # transform to cm
        self.pose[:,3] = new_hd
        