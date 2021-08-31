import numpy as np
from spikeA.Animal_pose import Animal_pose
from spikeA.Spike_train import Spike_train
from scipy.interpolate import interp1d
from scipy import ndimage

class Spatial_properties:
    """
    Class use to calculate the spatial properties of a single neuron.
    
    This can be used to calculate firing rate maps, information scores, grid scores, etc.
    
    Attributes:
        st = Spike_train object
        ap = Animal_pose object
    Methods:
        firing_rate_map_2d()
        
    """
    def __init__(self, spike_train=None, animal_pose=None):
        """
        Constructor of the Spatial_properties class
        """
        
#         if not isinstance(spike_train,Spike_train): 
#             raise TypeError("spike_train should be a subclass of the Spike_train class")
#         if not isinstance(animal_pose,Animal_pose): 
#             raise TypeError("animal_pose should be a subclass of the Animal_pose class")
            
        self.st=spike_train
        self.ap=animal_pose
        return
    
    def spike_position(self):
        """
        Method to calculate the position of each spike of the Spike_train object.
        
        Return
        self.spike_posi
        """
        # calculate the interpolatation function for x and y position data
        self.fx = interp1d(self.ap.pose[:,0], self.ap.pose[:,1], bounds_error=False)
        self.fy = interp1d(self.ap.pose[:,0], self.ap.pose[:,2], bounds_error=False)

        # create a numpy 2D array to store the spike position
        self.spike_posi = np.empty((self.st.st.shape[0],2))
        
        # get the position of the animal at each spike time
        self.spike_posi[:,0] = self.fx(self.st.st)
        self.spike_posi[:,1] = self.fy(self.st.st)
    
    def spike_head_direction(self):
        """
        Method to calculate the head direction of each spike of the Spike_train object.
        
        All the calculations are in radians
        
        The head-direction data are in radians. To do the interpolation, we transform in cos and sin components, interpolate and then back in radians.
        We transform in cos and sin components because averaging 0 and 360 gives 180, which is incorrect in the case of angles.
        
        Return
        self.spike_hd
        """
        
        # transform hd data to cos and sin components
        c = np.cos(self.ap.pose[:,4])
        s = np.sin(self.ap.pose[:,4])
        
        # fit
        fhdc = interp1d(self.ap.pose[:,0], c, bounds_error=False) # time, cos
        fhds = interp1d(self.ap.pose[:,0], s, bounds_error=False) # time, sin
        
        # interpolate
        ihdc = fhdc(self.st.st)
        ihds = fhds(self.st.st)
        
        # get radians from cos and sin
        self.spike_hd = np.arctan2(ihdc,ihds) 
        
        
        
    def firing_rate_head_direction_histogram(self,deg_per_bin=10, smoothing_sigma_deg = 20, smoothing=True):
        """
        Method of the Spatial_properties class to calculate the firing rate of a neuron as a function of head direction.
        
        If a compatible occupancy_hd histogram is not already present in the self.animal_pose object, one will be calculated.
        
        Calculations are all done in radians.
        
        Arguments:
        deg_per_bin: degrees per bins in the head-direction histogram
        smoothing_sigma_deg: standard deviation of the gaussian kernel used to smooth the firing rate head direction histogram
        smoothing: boolean indicating whether or not smoothing is applied
        
        Return:
        The Spatial_properties.firing_rate_head_direction_histo is set. It is a 1D numpy array containing the firing rate in Hz as a function of head direction.
        """
        self.hd_histo_deg_per_bin =deg_per_bin
        self.hd_histo_smoothing_sigma_deg = smoothing_sigma_deg
        self.hd_histo_smoothing = smoothing
        
      
        # create a new hd occupancy histogram
        self.ap.head_direction_occupancy_histogram(deg_per_bin =self.hd_histo_deg_per_bin, 
                                                 smoothing_sigma_deg = self.hd_histo_smoothing_sigma_deg, 
                                                 smoothing = True, zero_to_nan = True)
        
        self.spike_head_direction()
        
        ## calculate the number of spikes per bin in the histogram
        ## we use the bin edges of the occupancy histogram to make sure that the spike count histogram and hd occupancy histogram have the same dimension
        spike_count,edges = np.histogram(self.spike_hd, bins= self.ap.hd_occupancy_bins)
        
        self.spike_count_hd_histo=spike_count
        
        ## smooth the spike count array
        if smoothing:
            spike_count = ndimage.gaussian_filter1d(spike_count, sigma=self.hd_histo_smoothing_sigma_deg/self.hd_histo_deg_per_bin)
    
        ## get the firing rate in Hz (spike count/ time in sec)
        self.firing_rate_head_direction_histo_edges = self.ap.hd_occupancy_bins
        self.firing_rate_head_direction_histo = spike_count/self.ap.hd_occupancy_histogram
    
    def firing_rate_map_2d(self,cm_per_bin =2, smoothing_sigma_cm = 2, smoothing = True):
        """
        Method of the Spatial_properties class to calculate a firing rate map of a single neuron.
        
        If a compatible occupancy map is not already present in the self.animal_pose object, one will be calculated.
        
        Arguments
        cm_per_bin: cm per bins in the firing rate map
        smoothing_sigma_cm: standard deviation of the gaussian kernel used to smooth the firing rate map
        smoothing: boolean indicating whether or not smoothing should be applied to the firing rate map
        
        Return
        The Spatial_properties.firing_rate_map is set. It is a 2D numpy array containing the firing rate in Hz in a set of bins covering the environment.
        """
        # we could do check for valid value ranges
        self.map_cm_per_bin = cm_per_bin
        self.map_smoothing_sigma_cm = smoothing_sigma_cm
        self.map_smoothing = smoothing
        
        # create a new occupancy map
        self.ap.occupancy_map_2d(cm_per_bin =self.map_cm_per_bin, 
                                 smoothing_sigma_cm = self.map_smoothing_sigma_cm, 
                                 smoothing = True, zero_to_nan = True)
        
        ## get the position of every spike
        self.spike_position()
        
        ## calculate the number of spikes per bin in the map
        ## we use the bins of the occupancy map to make sure that the spike count maps and occupancy map have the same dimension
        spike_count,x_edges,y_edges = np.histogram2d(x = self.spike_posi[:,0], 
                                                     y= self.spike_posi[:,1],
                                                     bins= self.ap.occupancy_bins)
        
        # save this for later (e.g., in information_score())
        self.spike_count = spike_count
        
        ## smooth the spike count array
        if smoothing:
            spike_count = ndimage.filters.gaussian_filter(spike_count,
                                                          sigma=self.map_smoothing_sigma_cm/self.map_cm_per_bin)
    
        ## get the firing rate in Hz (spike count/ time in sec)
        self.firing_rate_map = spike_count/self.ap.occupancy_map
    
    def information_score(self):
        """
        Method of the Spatial_properties class to calculate the information score of a single neuron.
        
        The formula is from Skaggs and colleagues (1996, Hippocampus).
        
        Return
        Information score
        """      
        
        # need to check that we have a valid firing rate map already calculated
        # we need to check that the dimension of occ map and firing rate map are the same
        # we should not use smoothed firing rate maps
        
        # probability to be in bin i
        p = self.ap.occupancy_map/np.nansum(self.ap.occupancy_map)
        
        # firing rate in bin i
        v = self.firing_rate_map
        
        # mean rate is the sum of spike count / sum of occupancy, NOT the mean of the firing rate map bins
        mr = np.nansum(self.spike_count)/np.nansum(self.ap.occupancy_map)
        
        # when rate is 0, we get p * 0 * -inf, which should be 0
        # to avoid -inf * 0, we set the v==0 to np.nan
        v[v==0]=np.nan
        
        # following Skaggs' formula
        IS = np.nansum(p * v/mr * np.log2(v/mr))
        
        return IS
    
    def sparsity_score(self):
        """
        Method of the Spatial_properties class to calculate the sparsity score of a single neuron.
        
        Return
        Sparsity score
        """
        p = self.ap.occupancy_map/np.nansum(self.ap.occupancy_map)
        v = self.firing_rate_map
        return 1-(((np.nansum(p*v))**2)/np.nansum(p*(v**2)))
        

    
