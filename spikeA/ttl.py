import numpy as np

"""
This contains a function that detects the ttl pulse time (the one and off). 

Arguments:
ttl_data: an np.array that contains the ttl signals

Return a 2D array of all the ups and downs in the input data
"""
def detectTTL(self, ttl_data = None):
        """
        A function to detect ttl up and down edges
        
        Arguments:
        channel = ttl channel, by default the last channel
        start_sample = the first sample to read
        end_sample = the last sample to read
        
        Return a 2D array of all the ups and downs in the selected time window

        """
        diff = np.diff(ttl_data)
        edge = np.where(diff!=0)
        ttl = edge.reshape((int(len(edge)/2),2))
        
        return ttl
    
    