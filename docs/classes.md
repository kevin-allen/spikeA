# Main classes in spikeA

We will organize the code of spikeA in a set a classes. 

## Spike_train

Contain the spike time of a single neurons. Methods for analyzing a single spike train are defined here.

## Intervals

Analysis are often limited to specific time intervals. We will place the code that filter arrays for data within the intervals here.

## Session

This represent the recorded data from one animal during one session. 
What defines a session is that the data are all stored in the same directory with a specified structure.
The Session class can inspect the data directory to ensure that all files are there.
The session class coordinate how the data are loaded into memory.

## Cell_group

This class represents a group of cells, usually recorded at the same time in the same animal.
We will use this class to apply data processing steps to the Neurons in the Cell_group.
We could also use this class to do ensemble-level analysis.

## Neuron

This class represents a single neuron.

## Spatial_neuron

This class will inherit from neuron and have additional functions to study the spatial properties of the neuron.

## Animal_pose

This class contains the pose of the animal in time.
