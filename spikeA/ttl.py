import numpy as np

"""
This contains a function that detects the ttl pulse time (the one and off). 

Arguments:
ttl_data: an np.array that contains the ttl signals

Return a 2D array of all the ups and downs in the input data
"""
def detectTTL(ttl_data):
        """
        A function to detect ttl up and down edges
        
        Arguments:
        channel = np.array containing data from a ttl channel
        start_sample = the first sample to read
        end_sample = the last sample to read
        
        Return a 2D array of all the ups and downs in the selected time window
        """
        
        if not isinstance(ttl_data,np.ndarray):
            raise valueType("ttl_data should be a np.ndarray but is {}".format(type(ttl_data)))
        if ttl_data.ndim != 2:
            raise valueType("ttl_data should have 2 dimension but has {}".format(ttl_data.ndim))
        if ttl_data.shape[0] != 1:
            raise valueType("ttl_data first dimension should have a shape of 1 but has {}".format(ttl_data.shape[0]))
        
        # np.squeeze remove the first dimension
        diff = np.diff(np.squeeze(ttl_data))
        edge = np.where(diff!=0)[0]
        ttl = edge.reshape((int(len(edge)/2),2))
        
        return ttl
    
    