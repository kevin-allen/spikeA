import os.path
import os
import pandas as pd
import numpy as np
from datetime import datetime
from spikeA.Dat_file_reader import Dat_file_reader
from spikeA.Intervals import Intervals
import pickle

class Session:
    """
    Class containing information about a recording session.
    
    This class is very generic and should work with all types of recording sessions.
    
    We will derive more specific classes from it that will deal with the more specific stuff (klustakwik and kilosort sessions).
    
    Attributes:
        name: Name of the session. Usually used as the beginning of the file names. Format should be subject-date-time
        path: Directory path of the data for this session.
        subject: Name of the subect. This assumes that the session name is in the format subject-date-time
        session_date_time: Datetime of the session. This assumes that the session name is in the format subject-date-time
        file_base: path + name
        
    Methods:
        
    """
    def __init__(self,name, path):
        self.set_name_and_directories(name, path)
        self.find_session_data_type()
        
    def set_name_and_directories(self,name,path):
        """
        Method to set the name and path of the Session object
        """
        self.name = name
        self.path = os.path.normpath(path)
        self.subject = self.name.split("-")[0]
        self.session_dat_time = datetime.strptime(self.name.split("-")[1]+self.name.split("-")[2], '%d%m%Y%H%M')
        self.fileBase = self.path+"/"+name
        return
    
    def find_session_data_type(self):
        """
        Method to determine the data type of recording session
        Current possible data types are klustakwik or kilosort
        """
        
        if os.path.isfile(self.fileBase + ".clu"):
            self.data_type = "klustakwik"
        elif os.path.isfile(self.path +"/params.py"):
            self.data_type = "kilosort"
        else:
            raise ValueError("{}, unknown session data_type".format(self.name))
    
    def return_child_class(self):
        """
        Method that will return a Klustakwik_session or a Kilosort_session object depending on self.data_type
        
        This can be used to get the appropriate child object without having to figure it out during data analysis
        
        Example 1, single session
                
        from spikeA.Session import Session
        # create a Session object
        ses = Session(name="mn8578-30112021-0107",path="/adata/projects/autopi_mec/mn8578/mn8578-30112021-0107")
        # get a Kilosort_session object in this case
        ses = ses.return_child_class()
        
        Example 2, several sessions, here myProject.sessionList was an autopipy.Project object
        from spikeA.Session import Session
        # first create a list of spikeA.Sessions objects
        sSessions = [ Session(ses.name,ses.path) for ses in myProject.sessionList ] # spikeA.Session object
        # then get the right child object (Kilosort_session or Klustakwik_session) for each spikeA.Session object
        sSessions = [ ses.return_child_class() for sSes in sSessions ] # spikeA session object
        """
        
        if self.data_type == "klustakwik":
            return Klustakwik_session(self.name,self.path)
        elif self.data_type == "kilosort":
            return Kilosort_session(self.name,self.path)
    
    def session_environment_trial_data_frame(self):
        """
        Method to get the trial time as a pandas DataFrame. 
        Compare to using self.desen, it merges adjacent trials in time that are in the same environment.
        For examples: circ80 circ80 rest autopi autopi rest circ80 becomes circ80 rest autopi rest circ80

        You can use it to get the intervals in a given environment if you want adjacent trials in the same environment to be merged.
        
        Returns 
        pandas DataFrame with session, environment, no, trialCode, startTime, endTime and duration columns
        """
        envSeen = []
        envList = []
        trialCodeList = []
        envCountList = []
        envDuration = []
        start = []
        end = []
        prevEnv = ""
        prevIndex = -1
        for i, env in enumerate(self.desen):
            if env != prevEnv: # we have a new environment

                index = envSeen.count(env)
                trialCodeList.append(env+"_{}".format(index))

                envList.append(env)
                envCountList.append(index)

                envDuration.append(self.trial_intervals.inter[i][1]-self.trial_intervals.inter[i][0])
                prevEnv=env
                envSeen.append(env)
                start.append(self.trial_intervals.inter[i][0])
                end.append(self.trial_intervals.inter[i][1])
            else : 
                envDuration[-1]= envDuration[-1]+self.trial_intervals.inter[i][1]-self.trial_intervals.inter[i][0]
                end[-1] = self.trial_intervals.inter[i][1]


        return pd.DataFrame({"session":self.name,
                             "environment": envList,
                             "no": envCountList,
                             "trialCode": trialCodeList,
                             "startTime": start,
                             "endTime": end,
                             "duration": envDuration})
    
    def __str__(self): 
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
   
        
        
        
class Klustakwik_session(Session):
    """
    Class containing information about a recording session in which tetrodes were used with Klustakwik
    
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
            raise ValueError("{}, length of desel is not matching the number of tetrodes".format(self.name))
            
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
        
        
        # code that was in the constructor that was moved here
        # it was not documented...
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
        
        
        
        
        
    def __str__(self): 
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))




class Kilosort_session(Session):
    """
    Class containing information about a recording session processed with Kilosort.
    
    The files and format expected are described here:
    https://phy.readthedocs.io/en/latest/sorting_user_guide/
    
    Attributes:
        n_channels: Number of channels
        n_shanks: Number of shanks
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
                            "setup": self.fileBase + ".setup",
                            "environmentFamiliarity": self.fileBase + ".environmentFamiliarity",
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
    
    def load_parameters_from_files(self, ignore_params=False):
        """
        Function to read session parameters from configuration files.
        
        The names of the files are in the self.file_names dictionary.
        """
        
        ## check that the directory exists
        if not os.path.isdir(self.path):
            raise IOError("directory {} does not exist".format(self.path))
    
        
        ## read the params file (if not ignored, then only n_channels and sampling_rate will be read from other files, see below; anyway dat_dtype and dat_offset is not used)
        if not ignore_params:
            if not os.path.isfile(self.file_names["params"]):
                raise IOError("{} file not found".format(self.file_names["params"]))    
            f = open(self.file_names["params"], "r")
            c = f.read().replace('\'','').split('\n')
            f.close()
        
            self.n_channels = int(c[1].split(" = ")[1])
            self.dat_dtype = c[2].split(" = ")[1]
            self.dat_offset = int(c[3].split(" = ")[1])
            self.sampling_rate = float(c[4].split(" = ")[1])
            
        # get the trial names from the par file
        if not os.path.isfile(self.file_names["par"]):
            raise IOError("{} file not found".format(self.file_names["par"]))    
        f = open(self.file_names["par"], "r")
        c = f.read().split('\n')
        f.close()

        # read n_channels and sampling_rate from other files
        if ignore_params:
            # number of channels is also written in par file (do not read from params)
            self.n_channels = int(c[0].split()[0])
            # read the sampling_rate file
            if not os.path.isfile(self.file_names["sampling_rate"]):
                raise ValueError("{} file not found".format(self.file_names["sampling_rate"]))
            self.sampling_rate = int(open(self.file_names["sampling_rate"]).read().strip())


        to_skip = int(c[2].split()[0]) # = number of shanks, skip shank configuration
        self.n_shanks = to_skip
        self.n_trials = int(c[3+to_skip]) # read the number of trials, begins after shank channel list
        #print("n_trials",self.n_trials)
        self.trial_names = c[to_skip+4:to_skip+4+self.n_trials]
            
            
        # read the desen file
        if not os.path.isfile(self.file_names["desen"]):
            raise IOError("{} file not found".format(self.file_names["desen"]))
        self.desen = open(self.file_names["desen"]).read().strip().split('\n')
        
        # read the desel file 
        if not os.path.isfile(self.file_names["desel"]):
            raise IOError("{} file not found".format(self.file_names["desel"]))
        self.desel = open(self.file_names["desel"]).read().strip().split('\n')
        
        # read the stimulation file
        if not os.path.isfile(self.file_names["stimulation"]):
            raise IOError("{} file not found".format(self.file_names["stimulation"]))
        self.stimulation = open(self.file_names["stimulation"]).read().strip().split('\n')
        
        # read the setup file
        if not os.path.isfile(self.file_names["setup"]):
            raise IOError("{} file not found".format(self.file_names["setup"]))
        self.setup = open(self.file_names["setup"]).read().strip().split('\n')
        
        # read the environmentFamiliarity file
        if not os.path.isfile(self.file_names["environmentFamiliarity"]):
            raise IOError("{} file not found".format(self.file_names["environmentFamiliarity"]))
        self.environmentFamiliarity = open(self.file_names["environmentFamiliarity"]).read().strip().split('\n')
        
        # read the px_per_cm file
        if not os.path.isfile(self.file_names["px_per_cm"]):
            raise ValueError("{} file not found".format(self.file_names["px_per_cm"]))
        # self.px_per_cm = float(open(self.file_names["px_per_cm"]).read().split('\n')[0]) # for one value only
        px_per_cm = open(self.file_names["px_per_cm"]).read().strip().split('\n') # is an array with either length 1 or length = number of trials
        
        if len(px_per_cm)==1:
            self.px_per_cm = float(px_per_cm[0])
        elif len(px_per_cm)==self.n_trials:
            self.px_per_cm = np.array([ float(p) for p in px_per_cm ])
        else:
            raise ValueError("px_per_cm is invalid ({}), length must be either 1 or equal to number of trials ({})".format(len(px_per_cm),self.n_trials))
            
        #~ print("self.px_per_cm",type(self.px_per_cm),self.px_per_cm)
        

        # checks: these 4 files must have exactly one line for each trial, so that the length must match n_trials
        if len(self.desen) != self.n_trials:
            raise ValueError("{}: Length of desen is not matching the number of trials ({})".format(self.name,self.n_trials))
        if len(self.environmentFamiliarity) != self.n_trials:
            raise ValueError("{}: Length of environmentFamiliarity is not matching the number of trials ({})".format(self.name,self.n_trials))
        if len(self.setup) != self.n_trials:
            raise ValueError("{}: Length of setup is not matching the number of trials ({})".format(self.name,self.n_trials))
        if len(self.stimulation) != self.n_trials:
            raise ValueError("{}: Length of stimulation is not matching the number of trials ({})".format(self.name,self.n_trials))
            
        # check: the electrode configuration file must have one line per electrode, so that the length must match n_shanks
        if len(self.desel) != self.n_shanks:
            raise ValueError("{}: Length of desel is not matching the number of shanks".format(self.name))
        
        
        self.file_names["dat"] = [self.path+"/"+t+".dat" for t in self.trial_names]
        # self.dat_file_names is depreciated, use self.file_names["dat"] instead
        self.dat_file_names = [self.path+"/"+t+".dat" for t in self.trial_names]
        df = Dat_file_reader(file_names=self.dat_file_names,n_channels = self.n_channels)
        inter = df.get_file_intervals_in_seconds()
        self.trial_intervals = Intervals(inter)
        
        #####################################
        ## save self.trial_intervals as sessionIntervals.npy 
        ## for later use in the ses. directory
        ####################################
        
        fn = os.path.join(self.path, "sessionIntervals.npy")
        if not os.path.exists(fn):
            with open(fn, 'wb') as f:
                pickle.dump(self.trial_intervals, f)
        else:
            with open(fn, 'rb') as f:
                self.trial_intervals = np.load(f,allow_pickle=True)
        
        # load times collected externally
        times_fn = self.path + "/times.npy"
        if os.path.isfile(times_fn):
            self.log_times = np.load(times_fn)
        else:
            self.log_times = np.array([])

   
    ############
    ############ The code below does not really belong to the session class. It would make more sense to have this in the spike_wafeform class as it deals with spike waveforms.
    ############
        
    ##
    # Template Waveforms
    # - there exists for each template waveforms for each channel (see shape of self.templates = templates,timepoints,channels)
        
    def load_waveforms(self):
        """
        load the template waveforms from kilosorted files in that session
        """
        # load the template waveforms (3 dimensional array)
        ## for each template (wv_templates) there is a for each channel (wv_channels) the voltage for some sample time (wv_timepoints)
        self.templates = np.load(self.file_names["templates"])
        #print("templates.shape",self.templates.shape)
        wv_templates, wv_timepoints, wv_channels = self.templates.shape
        print("Templates:",wv_templates, ", timepoints:",wv_timepoints, ", Channels:",wv_channels)
        self.wv_channels = wv_channels
        
        # load the channel mapping
        self.channel_map = np.load(self.file_names["channel_map"]).flatten()
        # load the channel positions
        self.channel_positions = np.load(self.file_names["channel_positions"])

        
    def get_waveform(self, tmp, channel):
        """
        get the template waveform of template $tmp in channel $channel
        Returns: ( mapped channel name, waveform of that template in that specific channel )
        """        
        template_waveforms = self.get_waveforms(tmp)
        return ( self.channel_map[channel] , template_waveforms[:,channel] )
        
    def get_waveforms(self, tmp):
        """
        get the template waveforms of template $tmp in all channels
        Returns: ( waveforms of that template )
        """
        if not (0 <= tmp < len(self.templates)):
            raise ValueError("invalid template: {} / templates: {}".format(tmp, len(self.templates)))
        
        template_waveforms = self.templates[tmp]
        return template_waveforms
    
    def get_waveform_from_cluster(self, clu, channel):
        """
        get the waveform of cluster $clu in channel $channel
        Returns: ( mapped channel name, waveform of that cluster in that specific channel (calculated as weighted average from spike split/merge procedure) )
        """        
        #~print("get_waveform_from_cluster",clu,", ch:",channel)
        templates, weights = self.decompose_cluster(clu)
        #~print("templates, weights =",dict(zip(templates, weights)))
        waveform = np.sum([ weight * self.get_waveform(template, channel)[1] for template, weight in zip(templates, weights) ], axis=0)
        #~print("waveform",waveform.shape)
        return ( self.channel_map[channel] , waveform )

    
    ##
    # conversion: Template -> Cluster
    # - find the difference in templates & clusters after Phy post-processing
    
    def load_templates_clusters(self):
        # spike templates
        self.st = np.load(self.file_names["spike_templates"]).flatten() # np.load(data_prefix + "spike_templates.npy")[:,0]
        # spike clusters
        self.sc = np.load(self.file_names["spike_clusters"]).flatten() # np.load(data_prefix + "spike_clusters.npy")
        # check
        if len(self.st) != len(self.sc):
            raise ValueError("the length of spike_templates and spike_clusters should be the same but are {} / {}".format(len(self.st),len(self.sc)))
        # set list with all cluster ids
        self.clusterids = np.unique(self.sc).flatten()
        print("Loaded templates-clusters-map, spikes:", len(self.st),", clusters:",len(self.clusterids))

    # decompose cluster into templates
    def decompose_cluster(self, c):
        """
        phy split/merge leads to cluster = sum of templates. Function to decompose a cluster in its templates
        Returns: templates with its weights (proportion of number of spikes)
        """
        if not c in self.clusterids:
            raise ValueError("invalid cluster: {} from {} clusters".format(clu,len(self.clusterids)))
        
        s_ind = np.where(self.sc==c) # get spikes associated to that cluster $c
        s_templates = self.st[s_ind] # get templates associated to these spikes
        # now you have: cluster -> spikes -> templates, and thus a mapping from the cluster $c to the templates on which the corresponding spikes were detected
        unique, counts = np.unique(s_templates, return_counts=True) # get the distribution of the templates from the cluster $c
        weights = counts / np.sum(counts) # normalize
        return unique, weights

    """
    # you could run this to decompose each cluster in its templates
    for c in np.unique(sc):  ## self.clusterids
        print("cluster:",c)
        unique, weights = decompose_cluster(c)
        print(dict(zip(unique, weights)))
        print("")
    """
    
    
    
    ##
    # Channel assignments
        
    def init_shanks(self):
        """
        loads the shanks from the channel positions
        """
        # get shanks (assume x coordinate in channel_position) of channels
        self.shanks_all = np.unique(self.channel_positions[:,0])
        if len(self.shanks_all) != self.n_shanks:
            raise ValueError("Error in number of shanks! Check par/desel file (found {}) and kilosort/phy channel config (found {}).".format(self.n_shanks, len(self.shanks_all)))
        print("Init shanks:", len(self.shanks_all))
            
            
    def get_channels_from_waveforms(self, waveforms, cnt = 5, method="ptp"):
        """
        get $cnt channels with highest peak-to-peak amplitude or sum of voltages (method) in the waveforms
        $waveforms is 2D array of shape timepoints,channels
        Returns: array with channel ids with highest amplitude of length $cnt
        """
        
        if method=="ptp":
            amps = np.ptp(waveforms,axis=0) # method peak-to-peak (get peak-to-peak amplitude for each channel)
        elif method=="sum":
            amps = np.sum(np.abs(waveforms),axis=0) # method sum of voltage
        else:
            raise ValueError("invalid method provided to get amplitudes")
        
        
        #~print("get_channels_from_waveforms",waveforms.shape)
        
        channel_amps = np.array([range(self.wv_channels), amps]).T # table: channel id, amplitude
        #~print("channel_amps",channel_amps.shape)
        channel_amps = np.flip(sorted(channel_amps, key=lambda x: x[1]), axis=0) # sort by amplitude, descending (flip axis 0)
        channels_with_highest_amp = channel_amps[:cnt,0] # select first $cnt channels
        channels = channels_with_highest_amp.astype(int) # integer list
        return(channels) # the enumerated (non-translated, i.e. not mapped) channel ids



    def get_channels_from_template(self, tmp, cnt = 5, method="ptp"):
        """
        get $cnt channels with highest peak-to-peak amplitude in template $tmp
        Returns: array with channel ids with highest amplitude of length $cnt
        """
        ## template -> waveforms -> channels
        
        # first, get the waveforms of that template
        template_waveforms = self.get_waveforms(tmp) # = self.templates[tmp]
        # second, get the channels of that waveform
        return self.get_channels_from_waveforms(template_waveforms, cnt, method)
    
    def get_waveforms_from_cluster(self, clu):
        """
        get waveforms on all channels from cluster $clu
        """
        # transpose the result to maintain the shape: shape of waveforms = timepoints * channels
        waveforms = np.transpose([ self.get_waveform_from_cluster(clu, channel)[1] for channel in range(self.wv_channels) ])
        return waveforms
    
    
    def get_channels_from_cluster(self, clu, cnt = 5, method="ptp"):
        """
        get $cnt channels with highest peak-to-peak amplitude in cluster $clu
        Returns: array with channel ids with highest amplitude of length $cnt
        (This is a key function in the entire analysis. It uses many other functions to first collect the cluster's waveform by the templates' waveforms)
        """
        ## cluster -> waveforms -> channels
        
        # first, get the waveforms of that cluster        
        #~print("get_channels_from_cluster",clu)
        cluster_waveforms = self.get_waveforms_from_cluster(clu)
        #~print("cluster_waveforms",cluster_waveforms.shape)
        #~print(cluster_waveforms)
        # second, get the channels of that waveform
        return self.get_channels_from_waveforms(cluster_waveforms, cnt, method)

    
    def get_active_shanks(self, channels):
        """
        get information about shanks with these channels
        returns: shanks by name, by index, electrode locations (should be unique, len==1)
        """
        active_shanks = np.unique(self.channel_positions[channels][:,0]) # list of active shanks (for each channel, get its shank, list every occuring shank once)
        shanks_arr = [ shank in active_shanks for shank in self.shanks_all ] # boolean list with the information if shank is active
        electrodes = list(np.unique(np.array(self.desel)[shanks_arr])) # filter relevant electrode location / where shanks_arr==1
        return shanks_arr, active_shanks, electrodes
    

    ##
    # Environments by and from Trials
    
    def en2details(self,en):
        """
        converts the environment string (like "sqr-70_black_cue-W") to shape, diameter, color, cue-card position of arena
        """
        
        if en=="sqr70":
            return ("square",70.,None,None)
        if en=="sqr100":
            return ("square",100.,None,None)
        if en=="circ80":
            return ("circle",80.,None,None)
        if en=="rb":
            return ("rb",25.,None,None)
        
        nones = [None]*100
        typ,color,cue, *_ = en.split('_',2) + nones
        if typ=="rb":
            typ_shape = "sqr"
            diam = 25 # default restbox length = 25cm
        else:
            typ_shape, typ_diam = typ.split('-',1)
            diam = float(typ_diam)

        shape_dict = {'sqr':'square', 'circ':'circle'}
        shape = shape_dict.get(typ_shape, "unknown")

        return shape,diam,color,cue
    
    
    def session_trials(self):
        return [ (tn,su,en,self.en2details(en),ef,iv) for tn,su,en,ef,iv in zip(self.trial_names, self.setup, self.desen, self.environmentFamiliarity, self.trial_intervals.inter) ]
    
        
        
    def __str__(self): 
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
