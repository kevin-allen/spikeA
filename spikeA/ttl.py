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
        ttl_data: 1D np.array containing the data of a channel used for TTL
        
        Return a tuple of length 2 containing 1D arrays for the up and down edges of the TTL signal.  A tuple is returned instead of a 2D array because there might be more ups than downs in the signal
        """
        
        if not isinstance(ttl_data,np.ndarray):
            raise valueType("ttl_data should be a np.ndarray but is {}".format(type(ttl_data)))
        if ttl_data.ndim != 2:
            raise valueType("ttl_data should have 2 dimension but has {}".format(ttl_data.ndim))
        if ttl_data.shape[0] != 1:
            raise valueType("ttl_data first dimension should have a shape of 1 but has {}".format(ttl_data.shape[0]))
        
        # np.squeeze remove the first dimension
        diff = np.diff(np.squeeze(ttl_data))
        ups = np.where(diff>0)[0]
        downs = np.where(diff<0)[0]
        
        return ups,downs
    
    