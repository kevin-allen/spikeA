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
    
    def __init__(self,stl):
        """
        We create a list of Neuron object and set the spike trains of the neurons using a Spike_train_loader object. 
        
        Arguments:
        stl: Spike_train_loader object 
        
        # This function creates a list of Neurons and set the spike train object using the spike_train_loader object
        """
        if not isinstance(stl,Spike_train_loader):
            raise TypeError("stl should be a Spike_train_loader object but is {}".format(type(stl)))
        
        ## create a list of neurons
        ## use a list comprehension, use the stl.clu_ids to set the name of the neurons
        self.neuron_list=[Neuron(name=str(clu)) for clu in stl.clu_ids]
        
        ## set the spike_train objects of your neurons
        for i,n in enumerate(self.neuron_list):
               n.set_spike_train(st=stl.spike_times[i])
        

