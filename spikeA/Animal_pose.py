import numpy as np

class Animal_pose:
    """
    Class containing information about the pose (position and orientation) of an animal in time
    
    The position is in x,y,z and the orientation in Euler angles (yaw,pitch,roll). 
    When not available, data are set to np.NAN
    
    To ease computations, the time intervals between data point is kept constant.
    
    
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
    
    def pose_from_positrack_file(self,positrack_file_name, session, ttl_pulse_channel):
        """
        Method to calculute pose at fixed interval from a positrack file.
        
        Arguments
        positrack_file_name:
        dat_sampling_rate: sampling rate of the raw electrophysiological signal (used to synchronize pose to ephys recordings)
                
        Return
        The Animal_pose object has time and pose arrays set
        """
        
        #check if positrack file is there
        # read positrack file
        
        # get ttl channel from recording
        # detect ttl pulses
        
        # check that the number of pulses match
        #
        
        pass