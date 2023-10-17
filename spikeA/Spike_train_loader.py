"""
File containing the definition of the Spike_train_loader class
"""
import numpy as np
import pandas as pd
from scipy import stats
import os.path
from spikeA.Session import Kilosort_session
from spikeA.Session import Klustakwik_session


class Spike_train_loader:
    """
    Class use to load the spike train from files
    
    This class exists because spike trains might be stored differently depending on the specific recording systems and clustering programs used in different experiment.
    This class will take care of loading the spike trains and transform them in a standard representation that can be used in spikeA.
       
    Once loaded, the spike trains should be stored as 2 python lists: spike_times, clu_id
    
    Attributes:
    
    spike_times: A list of 1D np.arrays, one per neuron. It contains the spike times in seconds
    clu_ids: A list of items (usually integers) representing the cluster identification for each neuron
    sampling_rate: The sampling rate to transform some spike time inputs (from Klustakwik) from sample number to seconds
    
    
    """
    def __init__(self,sampling_rate=20000):
        """
        Constructor of the Spike_train_loader class

        Arguments:
        sampling_rate: number of samples per seconds
        """
        self.sampling_rate = sampling_rate       
    
    
    def load_session_spike_trains(self,session, verbose=False):
        """
        Load spikes from recording session. 
        
        Arguments
        session: a spikeA.Klustakwik_session or a spikeA.Kilosort_session object
        """
        
        if isinstance(session,Klustakwik_session):
            if verbose:
                print("load klustakwik spikes")
            self.load_spike_train_klustakwik(session, from_numpy_files = True)
        elif isinstance(session,spikeA.Session.Kilosort_session) or isinstance(session,Kilosort_session):
        #elif isinstance(session,Kilosort_session):    
            if verbose:
                print("load kilosort spikes")
            self.load_spike_train_kilosort(session, only_good = True)
        else: 
            raise TypeError("session should be a spikeA.KlustaKwik_session or a spikeA.Kilosort_session but is a {}".format(type(session)))
            
    
    
    def load_spike_train_klustakwik(self, klustakwik_ses, from_numpy_files = True):
        """
        Load the spike train for a recording session processed with klustakwik. 
        
        
        Arguments
        klustakwik_ses: a Klustakwik_session object
        
        Return
        The values are stored in self.clu_id, self.spike_times, 
        """
        if not isinstance(klustakwik_ses,Klustakwik_session):
            raise TypeError("klustakwik_ses should be a KlustaKwik_session but is {}".format(type(klustakwik_ses)))
        
        
        if from_numpy_files == True:
            if  os.path.isfile(klustakwik_ses.path+"/"+"spike_times.npy") and os.path.isfile(klustakwik_ses.path+"/"+"spike_times.npy"):
                self.load_spike_train_klustakwik_np_array(klustakwik_ses,verbose=False)
            else :
                print("npy files missing, loading from res and clu files")
                self.load_spike_train_klustakwik_res_clu_files(klustakwik_ses,verbose=False)
        else:
            self.load_spike_train_klustakwik_res_clu_files(klustakwik_ses,verbose=False)
            
        
    def load_spike_train_klustakwik_res_clu_files(self, klustakwik_ses, verbose=False):
        """
        Load the spike trains from res and clu files
        Also save the spike_times and spike_clusters as numpy array for faster loading.
        
        Reading from clu and res file is very slow. 
        That is why I save that data as numpy arrays.
        
        """
        if verbose:
            print("Reading",klustakwik_ses.file_names['res'])
        
        if not os.path.isfile(klustakwik_ses.file_names['res']):
            raise IOError("{} file not found".format(klustakwik_ses.file_names['res']))
        spike_times = np.squeeze(np.loadtxt(klustakwik_ses.file_names['res'], dtype=np.uint64)/self.sampling_rate).flatten()

        
        if verbose:
            print("Reading",klustakwik_ses.file_names['clu'])
        
        if not os.path.isfile(klustakwik_ses.file_names['clu']):
            raise IOError("{} file not found".format(klustakwik_ses.file_names['clu']))
        tmp = np.loadtxt(klustakwik_ses.file_names['clu'], dtype = np.int32).flatten()
        spike_clusters = tmp[1:]

        # check that the res and clu data have the same length
        if spike_clusters.shape[0] != spike_times.shape[0]:
            raise ValueError("the shape of spike_clusters and spike_times should be the same but are {} {}".format(spike_clusters.shape[0],spike_times.shape[0]))

        # save spikes as numpy array for faster loading during future analysis
        if verbose:
            print("Saving", klustakwik_ses.path+"/"+"spike_times")
        np.save(klustakwik_ses.path+"/"+"spike_times", spike_times)
        if verbose:
            print("Saving", klustakwik_ses.path+"/"+"spike_clusters")
        np.save(klustakwik_ses.path+"/"+"spike_clusters", spike_clusters)
        
        # ## format the spike times so we get a list of arrays
        self.clu_ids = np.sort(np.unique(spike_clusters))
        self.spike_times = [ spike_times[spike_clusters==c] for c in self.clu_ids ]

        
    def load_spike_train_klustakwik_np_array(self, klustakwik_ses,verbose=False):
        """
        Load the spike train from a spike_times and spike_clusters numpy array.
        
        Arguments
        klustakwik_ses: a KlustaKwik_session object
        
        The values are stored in self.clu_id, self.spike_times, 
        """
        if not isinstance(klustakwik_ses,Klustakwik_session):
            raise TypeError("klustakwik_ses should be a KlustaKwik_session but is {}".format(type(klustakwik_ses)))
        
        if verbose:
            print("Reading", klustakwik_ses.path+"/"+"spike_times.npy")
        if not os.path.isfile(klustakwik_ses.path+"/"+"spike_times.npy"):
            raise IOError("{} file not found".format(klustakwik_ses.path+"/"+"spike_times.npy"))
        spike_times = np.load(klustakwik_ses.path+"/"+"spike_times.npy")
        
        
        if verbose:
            print("Reading", klustakwik_ses.path+"/"+"spike_clusters.npy")
        if not os.path.isfile(klustakwik_ses.path+"/"+"spike_clusters.npy"):
            raise IOError("{} file not found".format(klustakwik_ses.path+"/"+"spike_clusters.npy"))
        spike_clusters = np.load(klustakwik_ses.path+"/"+"spike_clusters.npy")
        
        
        if spike_clusters.shape[0] != spike_times.shape[0]:
            raise ValueError("the shape of spike_clusters and spike_times should be the same but are {} {}".format(spike_clusters.shape[0],spike_times.shape[0]))
        
        self.clu_ids = np.sort(np.unique(spike_clusters))
        self.spike_times = [ spike_times[spike_clusters==c] for c in self.clu_ids ]
    
      
    def generate_klustakwik_clu_res(self, rep=10,clu_list = [0,1,2]):
        """
        Function to generate fake klustakwik data for testing
        
        Arguments:
        rep: number of spikes per cluster
        clu_list: list of cluster id
        
        Return
        The content for the clu and res file in the format of the clu and res files generated by klustakwik
        Values returned as a tuple (clu,res)
        """
        clu = np.repeat(a = clu_list,repeats=rep)
        n_clusters = np.array(len(clu_list))
        n_spikes = n_clusters*rep
        clu = np.insert(clu,0,n_clusters)
        res = np.linspace(start = 0, stop = 20000, num = n_spikes,dtype=np.int64)
        print("Number of clusters: {}, number of spikes: {}".format(n_clusters, n_spikes))
        print(clu.shape,res.shape)
        return (clu,res)
    
    def save_klustakwik_clu_res_files(self,clu_file_name,res_file_name,clu,res):
        """
        Save a clu and res file in the klustakwik format. 
        You can use the output of generate_klustakwik_clu_res() to get some data to save.
        
        Arguments
        clu_file_name
        res_file_name
        clu : content for the clu file
        res: content for the res file
        
        """
        print("save {}".format(clu_file_name))
        np.savetxt(fname = clu_file_name,X = clu.astype(int), fmt='%i')
        print("save {}".format(res_file_name))
        np.savetxt(fname = res_file_name,X=res.astype(int), fmt="%i")
    
        
    def load_spike_train_kilosort(self, ks, only_good = True):
        """
        Load the spike train for a recording session processed with kilosort
        
        Arguments
        ks: a Kilosort_session object
        only_good: flag to define if only clusters marked as good should be loaded (default: True)
        
        The values are stored in self.clu_id, self.spike_times, 
        """
        if not isinstance(ks,Kilosort_session):
            raise TypeError("ks should be a Kilosort_session but is {}".format(type(ks)))
        
        
        if not os.path.isfile(ks.file_names["spike_times"]):
            raise IOError("{} file not found".format(ks.file_names["spike_times"]))
        spike_times = np.squeeze(np.load(ks.file_names["spike_times"])/self.sampling_rate).flatten()
        
        if not os.path.isfile(ks.file_names["spike_clusters"]):
            raise IOError("{} file not found".format(ks.file_names["spike_clusters"]))
        spike_clusters = np.load(ks.file_names["spike_clusters"]).flatten()
        
        if spike_clusters.shape[0] != spike_times.shape[0]:
            raise ValueError("the shape of spike_clusters and spike_times should be the same but are {} {}".format(spike_clusters.shape[0],spike_times.shape[0]))
        
        # read the cluster_group file
        if only_good:
            
            if not os.path.isfile(ks.file_names["cluster_group"]):
                raise IOError("{} file not found".format(ks.file_names["cluster_group"]))

            cluster_group = pd.read_csv(ks.file_names["cluster_group"], sep="\t")

            # get the clu id of "good" clusters
            g = cluster_group.group == "good"
            good_clusters = cluster_group.cluster_id[g].to_numpy()
            #print("Number of good clusters: {}".format(len(good_clusters)))

            ## only keep the spikes from good clusters
            g = np.isin(spike_clusters,good_clusters)
            spike_times = spike_times[g]
            spike_clusters = spike_clusters[g]
        
        ## format the spike times so we get a list of arrays
        self.clu_ids = np.sort(np.unique(spike_clusters))
        self.spike_times = [ spike_times[spike_clusters==c] for c in self.clu_ids ]