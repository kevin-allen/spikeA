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
        path: Directory path of the data for this session.
        subject: Name of the subect. This assumes that the session name is in the format subject-date-time
        session_date_time: Datetime of the session. This assumes that the session name is in the format subject-date-time
        file_base: path + name
        
    Methods:
        
    """
    def __init__(self,name, path):
        self.name = name
        self.path = os.path.normpath(path)
        self.subject = self.name.split("-")[0]
        self.session_dat_time = datetime.strptime(self.name.split("-")[1]+self.name.split("-")[2], '%d%m%Y%H%M')
        self.fileBase = self.path+"/"+name
        return

    
class Tetrode_session(Session):
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
                          "res": self.fileBase + ".res",
                          "px_per_cm": self.fileBase + ".px_per_cm"}
        
        
        myPath = self.path+'/'
        cluFiles = [myPath + f for f in os.listdir(myPath) if f.endswith('.clu.', 0, 25) & ~f.endswith('.bk')]
        cluFiles.sort()
        cluFiles
        self.Tetrode_index = {}
        cell_index = 2

        for index, filename in enumerate(cluFiles):
            Tet = 'Tet_' + filename.split('/')[-1].split('.')[-1]
            nCell = int(open(filename).readline().split('\n')[0])-1
            if nCell != 0:
                cell_ID = range(cell_index, cell_index + nCell)
            elif nCell == 0:
                cell_ID = range(0,0)
            self.Tetrode_index[Tet] = cell_ID    
            cell_index = cell_index + nCell
        pass
    
    def load_parameters_from_files(self):
        """
        Function to read session parameters from configuration files
        
        """
        # check if the par file is there
        if not os.path.isfile(self.file_names["par"]):
            raise ValueError("{} file not found".format(self.file_names["par"]))                  
        df = open(self.file_names["par"]).read().split('\n')                   
        # read the number of channels
        self.n_channels = int(df[0].split(' ')[0])
        # read the number of tetrodes
        self.n_tetrodes = int(df[2].split(' ')[0])
        # create a list of 1D array with the channels for each tetrode
        tmp = df[3:int(self.n_tetrodes)+3]
        self.tetrode_channels = [list(map(int, tmp[i].split(' ')))[1:] for i in range(0, len(tmp))]
        # read the number of trials
        self.n_trials = int(df[int(self.n_tetrodes)+3])
        # get a list of trial names
        self.trial_names = df[self.n_tetrodes+4:self.n_tetrodes+4+self.n_trials]
        # check if the desen file is there
        if not os.path.isfile(self.file_names["desen"]):
            raise ValueError("{} file not found".format(self.file_names["desen"]))
        # read the desen file
        self.desen = open(self.file_names["desen"]).read().split('\n')[:-1]
        # check if the desel file is there
        if not os.path.isfile(self.file_names["desel"]):
            raise ValueError("{} file not found".format(self.file_names["desel"]))
        # read the desel file
        desel = open(self.file_names["desel"]).read().split('\n')[:-1]
        if len(desel) == self.n_tetrodes:
            self.desel = desel
        else:
            raise ValueError("Length of desel is not matching the number of tetrodes")
        # check if the sampling_rate file is there
        if not os.path.isfile(self.file_names["sampling_rate"]):
            raise ValueError("{} file not found".format(self.file_names["sampling_rate"]))
        self.sampling_rate = int(open(self.file_names["sampling_rate"]).read().split('\n')[0])
        
        # check if the px_per_cm file is there
        if not os.path.isfile(self.file_names["px_per_cm"]):
            raise ValueError("{} file not found".format(self.file_names["px_per_cm"]))
        self.px_per_cm = float(open(self.file_names["px_per_cm"]).read().split('\n')[0])
        
        
        
        
    def __str__(self): 
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))




class Kilosort_session(Session):
    """
    Class containing information about a recording session processed with Kilosort.
    
    The files and format expected are described here:
    https://phy.readthedocs.io/en/latest/sorting_user_guide/
    
    Attributes:
        n_channels: Number of channels
        n_shanks: Number of tetrodes
        dat_files: List of dat files
        trial_df: pandas data frame with information about the trials (trial_name,environment,start_sample,end_sample,duration_sec)
    
    Methods
        load_parameters_from_files()
    
    Usage:
        path="/home/kevin/Documents/data/perez_session/jp5520-26092015-0108"
        name="jp5520-26092015-0108"
        ks = KilosortSession(name,path)
        print("This is the config.py file: ", ks.file_names["config"])
        ks.load_parameters_from_files()
    """
    
    def __init__(self,name,path):
        super().__init__(name, path) # call Session constructor
        
        # get a dictionnary containing the files with the session configuration
        self.file_names = { # files used in the Allen lab
                            "par":self.fileBase +".par",
                           "desen":self.fileBase +".desen",
                            "desel":self.fileBase +".desel",
                            "sampling_rate":self.fileBase +".sampling_rate_dat",
                            "px_per_cm": self.fileBase + ".px_per_cm",
                           # files created by kilosort
                            "params": self.path +"/params.py",
                            "amplitudes": self.path +"/amplitudes.npy",
                            "channel_map": self.path +"/channel_map.npy",
                           "channel_positions": self.path +"/channel_positions.npy",
                           "pc_features": self.path +"/pc_features.npy",
                           "pc_feature_ind": self.path +"/pc_feature_ind.npy",
                           "spike_templates": self.path +"/spike_templates.npy",
                           "spike_times": self.path +"/spike_times.npy",
                           "spike_clusters": self.path +"/spike_clusters.npy",
                           "cluster_group": self.path +"/cluster_group.tsv"}
    
    def load_parameters_from_files(self):
        """
        Function to read session parameters from configuration files.
        
        The names of the files are in the self.file_names dictionary.
        """
        
        ## read the params file
        if not os.path.isfile(self.file_names["params"]):
            raise IOError("{} file not found".format(self.file_names["params"]))    
        f = open(self.file_names["params"], "r")
        c = f.read().replace('\'','').split('\n')
        f.close()
        
        self.n_channels = int(c[1].split(" = ")[1])
        self.dat_dtype = c[2].split(" = ")[1]
        self.dat_offset = int(c[3].split(" = ")[1])
        self.sampling_rate = float(c[4].split(" = ")[1])
        
        # read the desen file
        if not os.path.isfile(self.file_names["desen"]):
            raise IOError("{} file not found".format(self.file_names["desen"]))
        self.desen = open(self.file_names["desen"]).read().split('\n')[:-1]
        # read the desel file 
        if not os.path.isfile(self.file_names["desel"]):
            raise IOError("{} file not found".format(self.file_names["desel"]))
        # read the desel file
        self.desel = open(self.file_names["desel"]).read().split('\n')[:-1]
        
        # get the trial names from the par file
        if not os.path.isfile(self.file_names["par"]):
            raise IOError("{} file not found".format(self.file_names["par"]))    
        f = open(self.file_names["par"], "r")
        c = f.read().split('\n')
        f.close()
        to_skip = int(c[2].split()[0])
        n_trials = int(c[3+to_skip])
        n_trials
        self.trial_names = c[to_skip+4:to_skip+4+n_trials]
        
        
    def __str__(self): 
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))