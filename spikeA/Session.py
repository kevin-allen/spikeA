import os.path
import os
import pandas as pd
import numpy as np
from datetime import datetime
class Session:
    """
    Class containing information about a recording session.
    
    This class is very generic and should work with all types of recording sessions.
    
    We will derive more specific class from it that will deal with the more specific stuff.
    
    Attributes:
        name: Name of the session. Usually used as the beginning of the file names. Format should be subject-date-time
        path: Directory path of the data for this session. Should not end with a /
        subject: Name of the subect. This assumes that the session name is in the format subject-date-time
        session_date_time: Datetime of the session. This assumes that the session name is in the format subject-date-time
        file_base: path + name
        
    Methods:
        
    """
    def __init__(self,name, path):
        self.name = name
        self.path = path
        self.subject = self.name.split("-")[0]
        self.session_dat_time = datetime.strptime(self.name.split("-")[1]+self.name.split("-")[2], '%d%m%Y%H%M')
        self.fileBase = path+"/"+name
        return

    
class TetrodeSession(Session):
    """
    Class containing information about a recording session in which tetrodes were used
    
    Attributes:
        n_channels: Number of channels
        n_tetrodes: Number of tetrodes
        clustered: Boolean indicating whether the session is clustered
        tetrode_locations: List of brain region, one per tetrodes
        dat_files: List of dat files
        trial_df: pandas data frame with information about the trials (trial_name,environment,start_sample,end_sample,duration_sec)
    
    Methods
        load_parameters_from_files()
    
    """
    def __init__(self,name,path):
        # call the super class
        pass
    
    def load_parameters_from_files():
        pass


class ProbeSession(Session):
    """
    Class containing information about a recording session in which Neuropixels or Neuronexus probes were used
    """
    pass