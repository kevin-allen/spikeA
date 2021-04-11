import unittest
import numpy as np
from spikeA.Spike_train import Spike_train

class TestSpikeTrain(unittest.TestCase):

    def setUp(self):
        self.spikes = Spike_train(name = "my spike train",st=np.arange(200),sampling_rate=20000)
        
    def test_n_spikes(self):
        self.assertTrue(self.spikes.n_spikes() == np.arange(200).shape[0])   
                     
if __name__ == "__main__" :
    unittest.main()
