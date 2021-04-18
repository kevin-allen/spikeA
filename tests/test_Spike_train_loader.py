import unittest
import numpy as np
import tempfile # to make a temporary directory with some fake data
from spikeA.Spike_train_loader import Spike_train_loader

class TestSpikeTrain(unittest.TestCase):

    def setUp(self):
        self.sampling_rate = 20000
        self.stl = Spike_train_loader(sampling_rate=self.sampling_rate)
        
    def test_load_spike_train_klustakwik(self):
        
        ## we create files in a temporary directory, os-independent way of doing this
        with tempfile.TemporaryDirectory() as tmpdirname:
            print('created temporary directory', tmpdirname)
            res_file_name = tmpdirname+"/res"
            clu_file_name = tmpdirname+"/clu"    
            
            ## generate fake spike data in the Klustakwik format 
            self.clu, self.res = self.stl.generate_klustakwik_clu_res()
            ## save it to temporary clu and res files
            self.stl.save_klustakwik_clu_res_files(clu_file_name,res_file_name,self.clu,self.res)

            ## load the spike data from files in the klustakwik format and refromat it
            self.stl.load_spike_train_klustakwik(clu_file_name,res_file_name)

        # the temporary files are deleted
        
        # we can now compare the information in the clu and res arrays with stl.clu_ids and stl.spike_times
        
        # we should have the same number of cluster
        self.assertTrue(self.stl.clu_ids.size == self.clu[0])

        # we should have the same number of spikes
        num_spikes = np.sum([a.size for a in self.stl.spike_times])
        self.assertTrue(num_spikes == self.res.size)
        
        # clu data are 1 data point longer than res 
        self.assertTrue(num_spikes+1 == self.clu.size)
        
        
                     
if __name__ == "__main__" :
    unittest.main()
