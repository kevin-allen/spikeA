"""
File containing the definition of the Spike_train_loader class
"""
import numpy as np
import pandas as pd
from scipy import stats

class Spike_train_loader:
    """
    Class use to load the spike train from files
    
    This class exists because spike trains might be stored differently depending on the specific recording systems and clustering programs used in different experiment.
    This class will take care of loading the spike trains and transform them in a standard representation that can be used in spikeA.
    We will write a method for the different system we encounter. New users with different system can just add a new method in this class.
    
    Once loaded, the spike trains should be stored as 2 python lists: spike_times, clu_id
    
    Attributes:
    
    spike_times: A list of 1D np.arrays, one per neuron. It contains the spike times in seconds
    clu_ids: A list of items (usually integers) representing the cluster identification for each neuron
    sampling_rate: The sampling rate to transform some spike time inputs (from Klustakwik) from sample number to seconds
    
    Methods:
    generate_klustakwik_clu_res()
    save_klustakwik_clu_res_files()
    load_spike_train_klustakwik()
    load_spike_train_from_files_klustakwik()
    format_klustakwik_data()
    """
    def __init__(self,sampling_rate=20000):
        """
        Constructor of the Spike_train_loader class

        Arguments:
        sampling_rate: number of samples per seconds
        """
        
        self.sampling_rate = sampling_rate           

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
        print(clu)
        print(res)
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
    
    def load_spike_train_klustakwik(self,clu_file_name, res_file_name):
        """
        Load spike trains from KlustaKwik output and store it in the correct format for spikeA
        
        Klustakwik format:
        The time values are stored as sample number in a .res file.
        The cluster id of each spike in the .res file is found in the .clu file
        The first number in the .clu file is the number of clusters in the data set.
        The other numbers in the .clu file is the clu_id of each spike.
        So the .clu file has one more data point than the .res file.
        
        Arguments
        clu_file_name: the name of the clu file
        res_file_name: the name of the res file
        
        Results
        The spike trains loaded is stored in self.clu_ids and self.spike_times 
        """
        clu,res = self.load_spike_train_from_files_klustakwik(clu_file_name,res_file_name)
        self.format_klustakwik_data(clu,res)
        
    
    def load_spike_train_from_files_klustakwik(self,clu_file_name,res_file_name):
        """
        Load the clu and res file that were saved by klustakwik
        
        Arguments
        clu_file_name: name of the clu file
        res_file_name: name of the res file
        
        Return
        The content of the clu and res files are returned by the function as a tuple (clu,res)
        """
        print("read ",clu_file_name)
        clu = np.loadtxt(clu_file_name,dtype=int)
        print("read ",res_file_name)
        res = np.loadtxt(res_file_name,dtype=int)
        return (clu,res)
  
    def format_klustakwik_data(self, clu,res):
        """
        Take the content of the klustakwik clu and res files and transform it into a clu_ids list and spike_times list

        Arguments
        clu: content of the clu file
        res: content of the res file
        
        The values are stored in self.clu_id and self.spike_times
        """
        # time in seconds
        res = res/self.sampling_rate
        n_clusters = clu[0]
        clu = clu[1:]
        print("Number of cluster:",n_clusters)
        print(clu.size,res.size)
        self.clu_ids = np.sort(np.unique(clu))
        self.spike_times = [ res[clu==c] for c in self.clu_ids ]
        