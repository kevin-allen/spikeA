import unittest
import numpy as np
from spikeA.Spike_train import Spike_train

class TestSpikeTrain(unittest.TestCase):

    def setUp(self):
        self.spikes = Spike_train("hey",start_time=0, end_time=40000)
        self.spikes.st = [np.array(range(10)) , np.array(range(20))]
        
    def test_n_spikes_per_cluster(self):
        self.assertTrue(np.all(self.spikes.n_spikes_per_cluster() == np.array([10,20])))
                     
if __name__ == "__main__" :
    unittest.main()
