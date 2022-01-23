import os.path
import os
import pandas as pd
import numpy as np
from datetime import datetime
from spikeA.Dat_file_reader import Dat_file_reader
from spikeA.Intervals import Intervals


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
                           "stimulation":self.fileBase + ".stimulation",
                          "sampling_rate":self.fileBase +".sampling_rate_dat",
                          "clu": self.fileBase + ".clu",
                          "res": self.fileBase + ".res",
                          "px_per_cm": self.fileBase + ".px_per_cm"}
        
        
        myPath = self.path+'/'
        cluFiles = [myPath + f for f in os.listdir(myPath) if f.endswith('.clu.', 0, 25) & ~f.endswith('.bk')]
        cluFiles.sort()
        cluFiles
        self.tetrode_index = {}
        cell_index = 2

        for index, filename in enumerate(cluFiles):
            Tet = 'Tet_' + filename.split('/')[-1].split('.')[-1]
            nCell = int(open(filename).readline().split('\n')[0])-1
            if nCell != 0:
                cell_id = range(cell_index, cell_index + nCell)
            elif nCell == 0:
                cell_id = range(0,0)
            self.tetrode_index[Tet] = cell_id
            cell_index = cell_index + nCell
        pass
    
    def load_parameters_from_files(self):
        """
        Function to read session parameters from configuration files
        
        """
        
        ## check that the directory exists
        if not os.path.isdir(self.path):
            raise IOError("directory {} does not exist".format(self.path))
    
        # check if the par file is there
        if not os.path.isfile(self.file_names["par"]):
            raise ValueError("{} file not found".format(self.file_names["par"]))                  
        df = open(self.file_names["par"]).read().split('\n')       
        
        # read the number of channels
        self.n_channels = int(df[0].split(' ')[0])
        
        # read the number of tetrodes
        self.n_tetrodes = int(df[2].split(' ')[0])
        
        # create a dictionary with the channels for each tetrode
        tmp = df[3:int(self.n_tetrodes)+3]
        tetrode_channels = [list(map(int, tmp[i].split(' ')))[1:] for i in range(0, len(tmp))]
        self.tetrode_channels = {}
        for i in range(self.n_tetrodes):
            self.tetrode_channels['Tet_'+str(i)] = tetrode_channels[i]
        
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
        
        self.dat_file_names = [self.path+"/"+t+".dat" for t in self.trial_names]
        df = Dat_file_reader(file_names=self.dat_file_names,n_channels = self.n_channels)
        inter = df.get_file_intervals_in_seconds()
        self.trial_intervals = Intervals(inter)
        
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
        dat_file_names
        trial_intervals: spikeA.Intervals for each .dat file (or trial)
         
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
                            "stimulation":self.fileBase +".stimulation",
                            "px_per_cm": self.fileBase + ".px_per_cm",
                           # files created by kilosort
                            "params": self.path +"/params.py",
                            "amplitudes": self.path +"/amplitudes.npy",
                            "channel_map": self.path +"/channel_map.npy",
                           "channel_positions": self.path +"/channel_positions.npy",
                           "pc_features": self.path +"/pc_features.npy",
                           "pc_feature_ind": self.path +"/pc_feature_ind.npy",
                           "spike_templates": self.path +"/spike_templates.npy",
                           "templates": self.path +"/templates.npy",
                           "spike_times": self.path +"/spike_times.npy",
                           "spike_clusters": self.path +"/spike_clusters.npy",
                           "cluster_group": self.path +"/cluster_group.tsv"}
    
    def load_parameters_from_files(self):
        """
        Function to read session parameters from configuration files.
        
        The names of the files are in the self.file_names dictionary.
        """
        
        ## check that the directory exists
        if not os.path.isdir(self.path):
            raise IOError("directory {} does not exist".format(self.path))
    
        
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
        self.desel = open(self.file_names["desel"]).read().split('\n')[:-1]
        #read the stimulation file
        if not os.path.isfile(self.file_names["stimulation"]):
            raise IOError("{} file not found".format(self.file_names["stimulation"]))
        self.stimulation = open(self.file_names["stimulation"]).read().split('\n')[:-1]
        
        # check if the px_per_cm file is there
        if not os.path.isfile(self.file_names["px_per_cm"]):
            raise ValueError("{} file not found".format(self.file_names["px_per_cm"]))
        self.px_per_cm = float(open(self.file_names["px_per_cm"]).read().split('\n')[0])
        
        
        # get the trial names from the par file
        if not os.path.isfile(self.file_names["par"]):
            raise IOError("{} file not found".format(self.file_names["par"]))    
        f = open(self.file_names["par"], "r")
        c = f.read().split('\n')
        f.close()

        to_skip = int(c[2].split()[0]) # = number of shanks, skip shank configuration
        self.n_trials = int(c[3+to_skip]) # read the number of trials
        #print("n_trials",self.n_trials)
        self.trial_names = c[to_skip+4:to_skip+4+self.n_trials]

        self.file_names["dat"] = [self.path+"/"+t+".dat" for t in self.trial_names]
        # self.dat_file_names is depreciated, use self.file_names["dat"] instead
        self.dat_file_names = [self.path+"/"+t+".dat" for t in self.trial_names]
        df = Dat_file_reader(file_names=self.dat_file_names,n_channels = self.n_channels)
        inter = df.get_file_intervals_in_seconds()
        self.trial_intervals = Intervals(inter)
        
        
    def load_waveforms(self):
        """
        load the template waveforms from kilosorted files in that session
        """
        # load the template waveforms (3 dimensional array)
        ## for each cluster (wv_clusters) there is a for each channel (wv_channels) the voltage for some sample time (wv_timepoints)
        self.templates = np.load(self.file_names["templates"])
        print("templates.shape",self.templates.shape)
        wv_clusters, wv_timepoints, wv_channels = self.templates.shape
        print("Clusters:",wv_clusters, ", timepoints:",wv_timepoints, ", Channels:",wv_channels)
        self.wv_channels = wv_channels
        
        # load the channel mapping
        self.channel_map = np.load(self.file_names["channel_map"]).flatten()
        # load the channel positions
        self.channel_positions = np.load(self.file_names["channel_positions"])

    def init_shanks(self):
        """
        loads the shanks from the channel positions
        """
        # get shanks (assume x coordinate in channel_position) of channels
        self.shanks_all = np.unique(self.channel_positions[:,0])

    def get_active_shanks(self, channels):
        """
        get information about shanks with these channels
        returns: shanks by name, by index, electrode locations (should be unique, len==1)
        """
        active_shanks = np.unique(self.channel_positions[channels][:,0])
        #shanks_arr = np.zeros(len(self.shanks_all))
        #shanks_arr[[list(self.shanks_all).index(shank) for shank in active_shanks]]=1
        shanks_arr = np.array([ 1 if self.shanks_all[i] in active_shanks else 0 for i in range(len(self.shanks_all)) ]) # indices of active_shanks in shanks_all
        electrodes = np.unique(self.desel[shanks_arr]) # filter relevant electrode location
        return shanks_arr, active_shanks, electrodes

    def get_channels_from_cluster(self, clu, cnt = 5):
        """
        get $cnt channels with highest peak-to-peak amplitude in cluster $clu
        Returns: array with channel ids with highest amplitude of length $cnt
        """
        # get peak-to-peak amplitude for each channel
        
        if not (clu < len(self.templates)):
            return([])
        
        template_cluster = self.templates[clu]
        amps = np.ptp(template_cluster,axis=0)
        channel_amps = np.array([range(self.wv_channels),amps]).T
        channel_amps = np.flip(sorted(channel_amps, key=lambda x: x[1]))
        channels_with_highest_amp = channel_amps[:cnt,1]
        channels = channels_with_highest_amp.astype(int)
        return(channels) # the enumerated (non-translated, i.e. not mapped) channel ids
    
    def get_waveform(self, clu, channel):
        """
        get the template waveform of cluster $clu in channel $channel
        Returns: ( mapped channel name, template of that cluster in that specific channel )
        """        
        template_cluster = self.templates[clu]        
        return ( self.channel_map[channel] , template_cluster[:,channel] )
        
        
    def __str__(self): 
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
