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
        clustered: Boolean indicating whether the session is clustered (is there a main clu file)
        tetrode_locations: List of brain region, one per tetrodes
        dat_files: List of dat files
        trial_df: pandas data frame with information about the trials (trial_name,environment,start_sample,end_sample,duration_sec)
    
    Methods
        load_parameters_from_files()
    
    Usage:
        path="/home/kevin/Documents/data/perez_session/jp5520-26092015-0108"
        name="jp5520-26092015-0108"
        ts = TetrodeSession(name,path)
        print("This is the .par file: ", ts.file_names["par"])
        ts.load_parameters_from_files()
    
    """
    def __init__(self,name,path):
        super().__init__(name, path) # call Session constructor
        
        # get a dictionnary containing the files with the session configuration
        self.file_names = {"par":self.fileBase +".par",
                          "desen":self.fileBase +".desen",
                          "desel":self.fileBase +".desel",
                          "sampling_rate":self.fileBase +".sampling_rate_dat",
                          "clu": self.fileBase + ".clu",
                          "res": self.fileBase + ".res"}
        
        pass
    
    def load_parameters_from_files(self):
        """
        Function to read session parameters from configuration files
        """
        # check if the par file is there
        if os.path.isfile(self.file_names.get("par")):
            print("par file exists")
            df = open(self.file_names.get("par")).read().split('\n')
        else:
            raise ValueError("par file not found")
        # read the number of channels
        self.nCh = int(df[0].split(' ')[0])
        # read the number of tetrodes
        self.nTet = df[2].split(' ')[0]
        # create a list of 1D array with the channels for each tetrode
        tmp = df[3:int(self.nTet)+3]
        self.TetCh = [tmp[i].split(' ')[1:] for i in range(0, len(tmp))]
        # read the number of trials
        self.nTrials = df[int(self.nTet)+3]
        # get a list of trial names
        self.Trial_names = df[int(self.nTet)+4:int(self.nTet)+4+int(self.nTrials)]
        # check if the desen file is there
        if os.path.isfile(self.file_names.get("desen")):
            print("desen file exists")
        else:
            raise ValueError("desen file not found")
        # read the desen file
        self.desen = open(self.file_names.get("desen")).read().split('\n')[:-1]
        # check if the desel file is there
        if os.path.isfile(self.file_names.get("desel")):
            print("desel file exists")
        else:
            raise ValueError("desel file not found")
        # read the desel file
        self.desel = open(self.file_names.get("desel")).read().split('\n')[:-1]
        # check if the sampling_rate file is there
        if os.path.isfile(self.file_names.get("sampling_rate")):
            print(".sampling_rate file exists")
        else:
            raise ValueError(".sampling_rate file not found")
        # read the sampling rate
        self.sampling_rate = open(self.file_names.get("sampling_rate")).read().split('\n')[0]


class NeuronexusProbeSession(Session):
    """
    Class containing information about a recording session with Neuronexus probes
    """
    pass

class NeuropixelsSession(Session):
    """
    Class containing information about a recording session with Neuropixels probes
    """
    pass