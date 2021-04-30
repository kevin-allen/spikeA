import numpy as np
from spikeA.Neuron import Neuron
from spikeA.Spike_train_loader import Spike_train_loader

class Cell_group:
    """
    Class dealing with a group of neurons recorded in the same session.
    
    This class makes it easier to apply the same analysis on a list of neurons.
    The Cell_group class can also been used to do analysis of simultaneously recorded cells. 
    
    
    Attributes:
        neuron_list: List of Neuron objects
    
    Methods:
        __init__
    
    """
    
    def __init__(self,stl=None):
        """
        We create a list of Neuron object and set the spike trains of the neurons using a Spike_train_loader object. 
        
        Arguments:
        stl: Spike_train_loader object 
        
        # This function creates a list of Neurons and set the spike train object using the spike_train_loader object
        """
        if stl is None:
            return # nothing to do
        
        if not stl isinstance(stl,spikeA.Spike_train_loader.Spike_train_loader):
            raise TypeError("stl should be a SpikeA.Spike_train_loader.Spike_train_loader object")
        
        ## create a list of neurons 
        ## use a list comprehension, use the stl.clu_ids to set the name of the neurons
        #self.neuron_list = [ ... for clu in stl.clu_ids ]
        ## set the spike_train objects of your neurons
        ## use a for loop on your neuron object, call the neuron_list[i].set_spike_train(st)
        #for n in self.neuron_list:
        #    ...
