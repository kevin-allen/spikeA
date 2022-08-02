import numpy as np
from spikeA.Animal_pose import Animal_pose
from spikeA.Spike_train import Spike_train
import spikeA.spatial_properties
from scipy.interpolate import interp1d
from scipy import ndimage
from scipy.ndimage import sum as ndi_sum
from scipy.ndimage import center_of_mass as ndi_center_of_mass
from scipy.stats import pearsonr
import math
import cv2
import itertools



class Spatial_properties:
    """
    Class use to calculate the spatial properties of a single neuron.
    
    This can be used to calculate firing rate maps, information scores, grid scores, etc.
    
    Attributes:
        st = Spike_train object
        ap = Animal_pose object
    Methods:
        firing_rate_map_2d()
        spatial_autocorrelation_map_2d()
        spatial_autocorrelation_field_detection(threshold, neighborhood_size)
        spatial_autocorrelation_field_detection_7(neighborhood_size)
        
        
    """

    
    def __init__(self, ses=None, spike_train=None, animal_pose=None):
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
    
    def mid_point_from_edges(self, edges):
        """
        Find the middle point of the edges of bins (output of np.histogram) and therefore reduce the number of edges by 1.
        
        Arguments:
        edges: np.array containing the edges from np.histogram()
        
        Returns:
        A np.array containing the midpoint of every bin stored in the variable "timestamp".
        """
        x = edges
        diff = x[1] - x[0]
        median = diff/2
        
        timestamp = x[:-1] + median
        
        return timestamp
    
    def spike_position(self):
        """
        Method to calculate the position of each spike of the Spike_train object.
               
        
        Return
        self.spike_posi
        """
        
        ## check that the Animal_pose and Spike_train object have the same intervals
        ## this check is required because if there are spikes at time points with invalid pose values, they will be interpolated to a valid value.
        if self.ap.intervals.inter.shape != self.st.intervals.inter.shape:
            raise ValueError("The intervals in the Animal_pose and Spike_train objects are not the same. Please make sure the intervals in the Animal_pose and Spike_train objects are the same before calling spike_position()")
        if np.any(self.ap.intervals.inter != self.st.intervals.inter):
            raise ValueError("The intervals in the Animal_pose and Spike_train objects are not the same. Please make sure the intervals in the Animal_pose and Spike_train objects are the same before calling spike_position()")
        
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
        self.spike_hd = np.arctan2(ihds,ihdc) 
        
        
        
    def firing_rate_head_direction_histogram(self, deg_per_bin=10, smoothing_sigma_deg=10, smoothing=True):
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
        self.hd_histo_deg_per_bin = deg_per_bin
        self.hd_histo_smoothing_sigma_deg = smoothing_sigma_deg
        self.hd_histo_smoothing = smoothing
        
      
        # create a new hd occupancy histogram
        self.ap.head_direction_occupancy_histogram(deg_per_bin = self.hd_histo_deg_per_bin, 
                                                 smoothing_sigma_deg = self.hd_histo_smoothing_sigma_deg, 
                                                 smoothing = smoothing, zero_to_nan = True)
        #print(self.ap.hd_occupancy_histogram)
        
        self.spike_head_direction()
        
        ## calculate the number of spikes per bin in the histogram
        ## we use the bin edges of the occupancy histogram to make sure that the spike count histogram and hd occupancy histogram have the same dimension
        spike_count,edges = np.histogram(self.spike_hd, bins= self.ap.hd_occupancy_bins)
        
        self.spike_count_hd_histo=spike_count
        
        ## smooth the spike count array
        if smoothing:
            spike_count = ndimage.gaussian_filter1d(spike_count, sigma=self.hd_histo_smoothing_sigma_deg/self.hd_histo_deg_per_bin,mode="wrap")
    
        ## get the firing rate in Hz (spike count/ time in sec)
        self.firing_rate_head_direction_histo_edges = self.ap.hd_occupancy_bins
        self.firing_rate_head_direction_mid_angles=self.mid_point_from_edges(self.firing_rate_head_direction_histo_edges)
        self.firing_rate_head_direction_histo = spike_count/self.ap.hd_occupancy_histogram
    
    
    def head_direction_score(self):
        """
        Method to calculate the mean direction and the mean vector length from the hd-rate histogram
        
        The stats are based on self.firing_rate_head_direction_histo
        
        To get valid scores (not np.nan), the sum of firing rates should be larger than 0 and there should be no np.nan in the firing_rate_head_direction_histo
        
        returns a tuple: mean_direction_rad, mean_direction_deg, mean_vector_length, peak_angle_rad, peak_rate
        """
        if not hasattr(self, 'firing_rate_head_direction_histo'):
            raise TypeError("You need to call spatial_properties.firing_rate_head_direction_histogram() before calling this function")
            
        # sum up all spikes
        sum_histo = np.sum(self.firing_rate_head_direction_histo)
        
        # if all rates are at 0 or some at np.nan, we can't calculate these scores reliably
        if sum_histo == 0.0 or np.isnan(sum_histo):
            self.hd_mean_vector_length= np.nan
            self.hd_peak_angle_rad = np.nan
            self.hd_peak_rate = np.nan
            self.hd_mean_direction_deg = np.nan
            self.hd_mean_direction_rad = np.nan
            return (self.hd_mean_direction_rad,self.hd_mean_direction_deg, self.hd_mean_vector_length, self.hd_peak_angle_rad, self.hd_peak_rate)
        
        
        # get midth of bins
        angles=self.mid_point_from_edges(self.firing_rate_head_direction_histo_edges)
        
        
        # get x and y component of each angle and multiply by firing rate
        x = np.cos(angles)*self.firing_rate_head_direction_histo
        y = np.sin(angles)*self.firing_rate_head_direction_histo
                
        
        # the angle is the arctan of x divided by y
        mean_direction = np.arctan2(np.sum(y),np.sum(x))
        
        # angle of the peak
        self.hd_peak_angle_rad = angles[np.argmax(self.firing_rate_head_direction_histo)]
        self.hd_peak_rate = np.nanmax(self.firing_rate_head_direction_histo)
        
        self.hd_mean_direction_deg = mean_direction*360/(2*np.pi)
        self.hd_mean_direction_rad = mean_direction
        
        #get mean vector length
        R = np.sqrt(np.sum(x)**2+np.sum(y)**2)
        self.hd_mean_vector_length = R/sum_histo
        self.hd_variance = (1. - self.hd_mean_vector_length) * 180.

        return (self.hd_mean_direction_rad,self.hd_mean_direction_deg, self.hd_mean_vector_length, self.hd_peak_angle_rad, self.hd_peak_rate)
    
    
    def shuffle_head_direction_score(self, iterations=500, deg_per_bin=10, smoothing_sigma_deg=10, smoothing=True, percentile=95):
        """
        Get a distribution of HD score that would be expected by chance for this neuron

        Argument:
        iterations: How many shufflings to perform
        deg_per_bin: cm per bin in the HD histogram
        smoothing_sigma_deg, smoothing: smoothing in the HD histogram
        percentile: percentile of the distribution of shuffled info scores that is used to get the significance threshold

        Return
        tuple: 
        0: 1D numpy array with the grid scores obtained by chance for this neuron
        1: significance threshold for HD score
        
        Example
        
        # get a neuron and set intervals
        n = cg.neuron_list[7]
        n.spike_train.set_intervals(aSes.intervalDict[cond])
        n.spatial_properties.ap.set_intervals(aSes.intervalDict[cond])

        # get the observed value for HD score
        n.spatial_properties.firing_rate_head_direction_histogram()  
        HDS = n.spatial_properties.head_direction_score()[2]

        # get the shuffled values for grid score
        shuHDS,threshold = n.spatial_properties.shuffle_head_direction_score(iterations=100,percentile=95)

        # plot the results for this neuron
        res = plt.hist(shuHDS,label="shuffled")
        ymax=np.max(res[0])
        plt.plot([threshold,threshold],[0,ymax],c="black",label="Threshold")
        plt.plot([HDS,HDS],[0,ymax],c="red",label="Observed")
        plt.xlabel("HD score")
        plt.ylabel("Count")
        plt.legend()
        plt.show()
        """
        
          
        # keep a copy of the pose that we started with
        pose_at_start = self.ap.pose.copy()
        
        # allocate memory for the shuffle data
        self.head_direction_shuffle=np.empty(iterations)
        
        
        for i in range(iterations):
            self.ap.roll_pose_over_time() # shuffle the position data 
            self.firing_rate_head_direction_histogram(deg_per_bin=deg_per_bin, smoothing_sigma_deg = smoothing_sigma_deg, smoothing=smoothing)  
            self.head_direction_shuffle[i] = self.head_direction_score()[2] # calculate the new HD score (vector length only) with the shuffled HD data
            
            self.ap.pose=pose_at_start # reset the pose to the one we started with

        # calculate the threshold
        self.head_direction_score_threshold =  np.percentile(self.head_direction_shuffle,percentile)
        
        
        return self.head_direction_shuffle, self.head_direction_score_threshold
    

    def head_direction_tuning_distributive_ratio(self):
        """
        Method to calculate the distributive ratio of the observed and predicted hd tuning curve. The aim is to distinguish between hd selectivity and spatial selectivity.
        See eLife 2018;7:e35949 doi: 10.7554/eLife.35949
        
        No arguments
        
        Return
        distributive ratio
        """
        
        if not hasattr(self, 'firing_rate_map'):
            raise ValueError('Call self.firing_rate_map_2d() before calling spatial_properties.head_direction_tuning_distributive_ratio()')
        if not hasattr(self.ap, 'hd_occupancy_histogram_per_occupancy_bin'):
            raise ValueError('Call ap.head_direction_occupancy_histogram_per_occupancy_bin() before calling spatial_properties.head_direction_tuning_distributive_ratio()')
        if not hasattr(self.ap, 'occupancy_map'):
            raise ValueError('Call ap.occupancy_map_2d() before calling spatial_properties.head_direction_tuning_distributive_ratio()')
        
        fr_map=self.firing_rate_map
        occ_map=self.ap.occupancy_map
        
        if not hasattr(self.ap, 'hd_occupancy_histogram_per_occupancy_bin'):
            self.ap.head_direction_occupancy_histogram_per_occupancy_bin()
        pose_hd_hist=self.ap.hd_occupancy_histogram_per_occupancy_bin
        
        if not hasattr(self, 'firing_rate_head_direction_histo'):
            self.firing_rate_head_direction_histogram()
        obs_tuning_curve=self.firing_rate_head_direction_histo
        
        deg_per_bin = self.ap.hd_occupancy_deg_per_bin
        
        pred_tuning_curve=np.asarray([np.nansum([fr_map[i,j]*pose_hd_hist[i,j,angle] for i,j in np.nditer(np.meshgrid(range(occ_map.shape[0]), range(occ_map.shape[1])))])/np.nansum(pose_hd_hist[:,:,angle], axis=(0,1)) for angle in range(int(360/deg_per_bin))])
        pred_tuning_curve[pred_tuning_curve==np.inf]=np.nan
        
        DR=np.nansum(np.abs(np.log((1+obs_tuning_curve)/(1+pred_tuning_curve))))/int(360/deg_per_bin)
        
        return DR
    
    
    
    def firing_rate_head_direction_histogram_binned(self, sub_intervals):
    

        hd_firing_all = []
        hd_mvl_all = []
        hd_mean_direction_rad_all = []
        hd_peak_angle_rad_all = []
        hd_peak_rate_all = []
        
        mean_firing_rate_all = []

        for sub_interval in sub_intervals:

            # reset and set intervals
            self.set_intervals(sub_interval)

            # calculate HD tuning curve
            self.firing_rate_head_direction_histogram(smoothing=False)
            angles = self.mid_point_from_edges(self.firing_rate_head_direction_histo_edges)
            hd_firing = self.firing_rate_head_direction_histo
            hd_mean_direction_rad, hd_mean_direction_deg, hd_mean_vector_length, hd_peak_angle_rad, hd_peak_rate = self.head_direction_score()

            hd_firing_all.append(hd_firing)
            hd_mvl_all.append(hd_mean_vector_length)
            hd_mean_direction_rad_all.append(hd_mean_direction_rad)
            hd_peak_angle_rad_all.append(hd_peak_angle_rad)
            hd_peak_rate_all.append(hd_peak_rate)
            
            mean_firing_rate_all.append(self.st.mean_firing_rate())
            
        return np.array(hd_firing_all), np.array(hd_mvl_all), np.array(hd_mean_direction_rad_all), np.array(hd_peak_angle_rad_all), np.array(hd_peak_rate_all), np.array(mean_firing_rate_all)
    



    
        
    
    def firing_rate_map_2d(self,cm_per_bin=2, smoothing_sigma_cm=2, smoothing = True, xy_range=None):
        """
        Method of the Spatial_properties class to calculate a firing rate map of a single neuron.
        
        If a compatible occupancy map is not already present in the self.animal_pose object, one will be calculated.
        
        Arguments
        cm_per_bin: cm per bins in the firing rate map
        smoothing_sigma_cm: standard deviation of the gaussian kernel used to smooth the firing rate map
        smoothing: boolean indicating whether or not smoothing should be applied to the firing rate map
        xy_range: 2D np.array of size 2x2 [[xmin,ymin],[xmax,ymax]] with the minimal and maximal x and y values that should be in the occupancy map. This can be used to set the firing rate map to a specific size. The default value is None, which means that the size of the occupancy map (and firing rate map) will be determined from the range of values in the Animal_pose object.
        
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
                                 smoothing = smoothing, zero_to_nan = True,xy_range=xy_range)
        
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
       
    def firing_rate_histogram(self,cm_per_bin=2, smoothing_sigma_cm=2,smoothing=True,x_range=None,linspace=False,n_bins = None):
        """
        Method of the Spatial_properties class to calculate a firing rate histogram (1D) of a single neuron.
        This is similar to firing_rate_map_2d, but only use the x position values of the self.animal_pose object.
        
        Arguments
        cm_per_bin: cm per bins in the firing rate map
        smoothing_sigma_cm: standard deviation of the gaussian kernel used to smooth the firing rate map
        smoothing: boolean indicating whether or not smoothing should be applied to the firing rate map
        x_range: 1D np.array of size 2 [xmin,xmax] with the minimal and maximal x to be included in the occupancy map. This can be used to set the firing rate map to a specific size. The default value is None, which means that the size of the occupancy histogram (and firing rate histogram) will be determined from the range of values in the Animal_pose (x) object.
        linspace: alternative way to create the binning, using np.linespace instaead of np.arange. Was introduced because I was getting inconsistencies in the number of bins when using np.arange. If linspace is true, cm_per_bin will not be used. Instead n_bins will be used.
        n_bins: if using linspace, this will be the number of bins in your histogram. If linspace is False, n_bins is not used.
        
        Return
        The Spatial_properties.firing_rate_histo is set. It is a 1D numpy array containing the firing rate in Hz in a set of bins covering the environment.
        """
        
        # we could do check for valid value ranges
        self.map_cm_per_bin = cm_per_bin
        self.map_smoothing_sigma_cm = smoothing_sigma_cm
        self.map_smoothing = smoothing
        
        # create a occupancy histogram
        self.ap.occupancy_histogram_1d(cm_per_bin =self.map_cm_per_bin, 
                                 smoothing_sigma_cm = self.map_smoothing_sigma_cm, 
                                 smoothing = smoothing, zero_to_nan = True,x_range=x_range, linspace = linspace,n_bins=n_bins)
        
        # this will work with the x and y data, but we will only use the x
        self.spike_position() 
        
        spike_count,x_edges = np.histogram(self.spike_posi[:,0],bins= self.ap.occupancy_bins)
        
        # save this for later 
        self.spike_count = spike_count
        
        ## smooth the spike count array
        if smoothing:
            spike_count = ndimage.gaussian_filter1d(spike_count,sigma=smoothing_sigma_cm/cm_per_bin,mode="nearest")
    
        ## get the firing rate in Hz (spike count/ time in sec)
        self.firing_rate_histo = spike_count/self.ap.occupancy_histo
        self.firing_rate_histo_mid = self.mid_point_from_edges(x_edges)

    
    def information_score_histogram(self):
        """
        Method of the Spatial_properties class to calculate the information score of a single neuron.
        
        The formula is from Skaggs and colleagues (1996, Hippocampus).
        
        You should have calculated firing_rate_histo without smoothing before calling this function
        
        Return
        Information score
        """      
        
        if not hasattr(self, 'firing_rate_histo'):
            raise ValueError('Call self.firing_rate_histogram() before calling self.information_score_histogram()')
        if self.map_smoothing == True:
            print("You should not smooth the firing rate map when calculating information score")
        
        if np.any(self.ap.occupancy_histo.shape != self.firing_rate_histo.shape):
            raise ValueError('The shape of the occupancy histogram should be the same as the firing rate histogram.')
        
        if np.nanmax(self.firing_rate_histo)==0.0: # if there is no spike in the map, we can't really tell what the information is.
            return np.nan
        
                
        # probability to be in bin i
        p = self.ap.occupancy_histo/np.nansum(self.ap.occupancy_histo)
        
        # firing rate in bin i
        v = self.firing_rate_histo.copy() # we need to make a copy because we will modify it a few lines below
        
        # mean rate is the sum of spike count / sum of occupancy, NOT the mean of the firing rate map bins
        mr = np.nansum(self.spike_count)/np.nansum(self.ap.occupancy_histo)
        
        # when rate is 0, we get p * 0 * -inf, which should be 0
        # to avoid -inf * 0, we set the v==0 to np.nan
        v[v==0]=np.nan
        
        # following Skaggs' formula
        IS = np.nansum(p * v/mr * np.log2(v/mr))
        
        return IS
    
    
    def information_score(self):
        """
        Method of the Spatial_properties class to calculate the information score of a single neuron.
        
        The formula is from Skaggs and colleagues (1996, Hippocampus).
        
        You should have calculated firing_rate_maps without smoothing before calling this function
        
        Return
        Information score
        """      
        
        if not hasattr(self, 'firing_rate_map'):
            raise ValueError('Call self.firing_rate_map_2d() before calling self.information_score()')
        if self.map_smoothing == True:
            print("You should not smooth the firing rate map when calculating information score")
        
        if np.any(self.ap.occupancy_map.shape != self.firing_rate_map.shape):
            raise ValueError('The shape of the occupancy map should be the same as the firing rate map.')
        
        if np.nanmax(self.firing_rate_map)==0.0: # if there is no spike in the map, we can't really tell what the information is.
            return np.nan
        
        # probability to be in bin i
        p = self.ap.occupancy_map/np.nansum(self.ap.occupancy_map)
        
        # firing rate in bin i
        v = self.firing_rate_map.copy() # we need to make a copy because we will modify it a few lines below
        
        # mean rate is the sum of spike count / sum of occupancy, NOT the mean of the firing rate map bins
        mr = np.nansum(self.spike_count)/np.nansum(self.ap.occupancy_map)
        
        # when rate is 0, we get p * 0 * -inf, which should be 0
        # to avoid -inf * 0, we set the v==0 to np.nan
        v[v==0]=np.nan
        
        # following Skaggs' formula
        IS = np.nansum(p * v/mr * np.log2(v/mr))
        
        return IS
    
    def shuffle_info_score(self, iterations=500,cm_per_bin=2,percentile=95):
        """
        Get a distribution of information score that would be expected by chance for this neuron

        Argument:
        iterations: How many shufflings to perform
        cm_per_bin: cm per bin in the firing rate map
        percentile: percentile of the distribution of shuffled info scores that is used to get the significance threshold

        Return
        tuple: 
        0: 1D numpy array with the information scores obtained by chance for this neuron
        1: significance threshold for information score
        
        Example
        
        # get a neuron and set intervals
        n = ses.cg.neuron_list[7]
        n.spike_train.set_intervals(aSes.intervalDict[cond])
        n.spatial_properties.ap.set_intervals(aSes.intervalDict[cond])

        # get the observed value for information score
        n.spatial_properties.firing_rate_map_2d(cm_per_bin=2, smoothing=False)    
        IS = n.spatial_properties.information_score()

        # get the shuffled values for information score
        shuIS,threshold = n.spatial_properties.shuffle_info_score(iterations=100, cm_per_bin=2,percentile=95)

        # plot the results for this neuron
        res = plt.hist(shuIS,label="shuffled")
        ymax=np.max(res[0])
        plt.plot([threshold,threshold],[0,ymax],c="black",label="Threshold")
        plt.plot([IS,IS],[0,ymax],c="red",label="Observed")
        plt.xlabel("Information score")
        plt.ylabel("Count")
        plt.legend()
        plt.show()
        """
        
        # keep a copy of the pose that we started with
        pose_at_start = self.ap.pose.copy()
        
        self.spatial_info_shuffle=np.empty(iterations)
        for i in range(iterations):
            self.ap.roll_pose_over_time() # shuffle the position data 
            self.firing_rate_map_2d(cm_per_bin=cm_per_bin, smoothing=False) # calculate a firing rate map
            self.spatial_info_shuffle[i] = self.information_score() # calculate the IS from the new map
            self.ap.pose=pose_at_start

        # calculate the threshold
        self.spatial_info_score_threshold =  np.percentile(self.spatial_info_shuffle,percentile)
        
       
        return self.spatial_info_shuffle, self.spatial_info_score_threshold

    
    def sparsity_score(self):
        """
        Method of the Spatial_properties class to calculate the sparsity score of a single neuron.
        
        Return
        Sparsity score
        """
        if not hasattr(self, 'firing_rate_map'):
            raise ValueError('Call self.firing_rate_map_2d() before calling self.information_score()')
        if self.map_smoothing == True:
            print("You should not smooth the firing rate map when calculating information score")
        
        if np.any(self.ap.occupancy_map.shape != self.firing_rate_map.shape):
            raise ValueError('The shape of the occupancy map should be the same as the firing rate map.')
        
        p = self.ap.occupancy_map/np.nansum(self.ap.occupancy_map)
        v = self.firing_rate_map
        return 1-(((np.nansum(p*v))**2)/np.nansum(p*(v**2)))
        
        
    def find_field_pixels(self, p, field_pixels, rate_map, peak_rate, min_fraction_of_peak_rate):
        """
        Method of the Spatial_properties class to determine if adjacent pixels to a start pixel belong to the firing field in the firing rate map.
        
        This function is recursive. 
        It is called by detect_one_field() called by firing_rate_field_detection().
        Arguments:
        p: start pixel
        field_pixels: list containing the pixels belonging to the field (contains the start pixel at first and then the adjacent field pixels are appended)
        rate_map: copy of the firing rate map
        peak_rate: peak rate in the firing rate map
        min_fraction_of_peak_rate: threshold firing rate of a pixel to be considered a field pixel (as fraction of peak rate)
        Return
        The field pixels are appended to the field_pixels list
        """
        # check all 8 adjacent pixels
        y=p[0];x=p[1]
        adjacent_pixels=[(y+1,x),(y+1,x+1),(y,x+1),(y-1,x+1),(y-1,x),(y-1,x-1),(y,x-1),(y+1,x-1)]
        for p in adjacent_pixels:
            # to be considered a field pixel, a pixel should be wihin the range of the firing rate map and have a firing rate > min_fraction_of_peak_rate peak rate
            if p not in field_pixels and p[0]<rate_map.shape[0] and p[1]<rate_map.shape[1] and rate_map[p]>peak_rate*min_fraction_of_peak_rate:
                field_pixels.append(p)
                # use field pixel as new starting point to detect all pixels belonging to the same firing field
                self.find_field_pixels(p, field_pixels, rate_map, peak_rate, min_fraction_of_peak_rate)
        

    def detect_one_field(self, rate_map, fields, peak_rate, min_pixel_number_per_field, min_peak_rate, min_fraction_of_peak_rate):
        """
        Method of the Spatial_properties class to detect firing fields in the firing rate map.
        
        This function is recursive. 
        It is called by firing_rate_field_detection().
        Arguments:
        rate_map: copy of the firing rate map
        fields: the list to which detected fields are appended (empty at first)
        peak_rate: the max firing rate in the original firing rate map
        min_pixel_number_per_field: minimal pixel number so that a putative field is considered a field
        min_fraction_of_peak_rate: threshold firing rate of a pixel to be considered a field pixel (as fraction of peak rate)
        
        Return
        The fields are appended to the fields list. The fields list is returned.
        """
        # the start pixel should have the maximal firing rate in the map. Detected fields and attempted start pixels are removed from the map so that we get a new start pixel each time.
        peak_pixel_rate = np.nanmax(rate_map)
        # The start pixel should have a rate higher than min_peak_rate. If several pixels have the highest firing rate and fulfill this criterion, the first one is selected.
        if peak_pixel_rate > min_peak_rate:
            peak_pixel = np.where(rate_map == peak_pixel_rate)
            if peak_pixel[0].shape[0]>1:
                start_pixel=(peak_pixel[0][0], peak_pixel[1][0])
            else:
                start_pixel=peak_pixel
            field_pixels=[start_pixel]

            # determine if the adjacent pixels belong to the putative field
            self.find_field_pixels(start_pixel, field_pixels, rate_map, peak_rate, min_fraction_of_peak_rate)
            # set the start pixel to nan so that it will not be selected again as start pixel
            rate_map[start_pixel]=np.nan
            # a firing field must have a minimal number of pixels
            if len(field_pixels)>min_pixel_number_per_field:
                fields.append(field_pixels)
                # set all field pixels to nan so that they will not be assigned to other fields
                for p in field_pixels:
                    rate_map[p]=np.nan
            # check if there could be more fields (no more fields when all pixels are nan or have too low firing rate)
            if not all(all(np.isnan(rate_map[:,r])) for r in range(rate_map.shape[1])) or all(all(rate_map[:,r]<peak_rate*min_fraction_of_peak_rate) for r in range(rate_map.shape[1])):
                self.detect_one_field(rate_map, fields, peak_rate, min_pixel_number_per_field, min_peak_rate, min_fraction_of_peak_rate)
        return(fields)
    
    
    def firing_rate_map_field_detection(self, min_pixel_number_per_field=25, min_peak_rate=4, min_fraction_of_peak_rate=0.2, cm_per_bin=2):
        """
        Method of the Spatial_properties class to calculate the position and size of fields in the firing rate map.
        
        If a compatible firing rate map is not already present in the spatial_properties object, an error will be given.
        Arguments:
        min_pixel_number_per_field: minimal number of pixels so that the putative firing field will be appended to the fields list
        min_peak_rate: minimal firing rate in a field
        min_fraction_of_peak_rate: threshold firing rate of a pixel to be considered a field pixel (as fraction of peak rate)
        cm_per_bin: cm_per_bin as for calculation of firing rate map
        Return
        The Spatial_properties.firing_rate_map_field_size and Spatial_properties.firing_rate_map_fields are set.
        """
        ## check for firing rate map
        if not hasattr(self, 'firing_rate_map'):
            raise TypeError("Call spatial_properties.firing_rate_map_2d() before calling spatial_properties.firing_rate_map_field_detection()")
        
        # work on a copy of the firing rate map because fields will be set to nan
        rate_map = self.firing_rate_map.copy()
        # invalid pixels should be nan
        rate_map[rate_map==-1]=np.nan
        # create an empty list to which the detected fields will be appended
        fields = []
        # get the peak rate of the whole map
        peak_rate = np.nanmax(rate_map)
        # call the recursive function detect_one_field which will find all the fields
        fields = self.detect_one_field(rate_map, fields, peak_rate, min_pixel_number_per_field, min_peak_rate, min_fraction_of_peak_rate)
        self.firing_rate_map_fields = fields
        # calculate the field size in cm2
        if fields:
            self.firing_rate_map_field_size = [len(fields[i])*cm_per_bin**2 for i in range(len(fields))]
        else:
            self.firing_rate_map_field_size = []

    
    
    def spatial_autocorrelation_map_2d(self):
        """
        Method of the Spatial_properties class to calculate a spatial autocorrelation map of a single neuron.
        
        If a compatible firing rate map is not already present in the spatial_properties object, an error will be given.
        
        Arguments
        
        Return
        The Spatial_properties.spatial_autocorrelation_map is set. It is a 2D numpy array.
        """
        
        ## check for firing rate map
        if not hasattr(self, 'firing_rate_map'):
            raise TypeError("Call spatial_properties.firing_rate_map_2d() before calling spatial_properties.spatial_autocorrelation_map_2d()")
        
        ## check for smoothing
        if not self.map_smoothing:
            print("You should smooth the firing rate map when calculating autocorrelation in order to detect the fields")
        
        ## convert nan values to -1 for C function
        self.firing_rate_map[np.isnan(self.firing_rate_map)]=-1.0
        
        ## create an empty array of the appropriate dimensions to store the autocorrelation data
        auto_array = np.zeros((2*self.firing_rate_map.shape[0]+1,2*self.firing_rate_map.shape[1]+1))

        ## create the spatial autocorrelation calling a C function
        spikeA.spatial_properties.map_autocorrelation_func(self.firing_rate_map,auto_array)
        self.spatial_autocorrelation_map = auto_array

        
        
    def spatial_autocorrelation_field_detection(self, threshold = 0.1, neighborhood_size = 5):
        """
        Method to detect fields based on autocorrelation map
        
        Returns
        The list of peaks x,y        
        """
        
        ### calculate new autocorrelation map
        self.spatial_autocorrelation_map_2d()
            
        data = self.spatial_autocorrelation_map
        
        data_max = ndimage.filters.maximum_filter(data, neighborhood_size)
        maxima = (data == data_max)
        data_min = ndimage.filters.minimum_filter(data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0

        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        x, y = [], []
        for dy,dx in slices:
            x_center = (dx.start + dx.stop - 1)/2
            x.append(round(x_center))
            y_center = (dy.start + dy.stop - 1)/2    
            y.append(round(y_center))

        self.spatial_autocorrelation_field = (x,y)
    
    
    
    def spatial_autocorrelation_field_detection_7(self, neighborhood_size = 5):
        thresholds = np.linspace(0,0.1,100)
        numsofpeaks = [ len(self.spatial_autocorrelation_field_detection(threshold, neighborhood_size)[0]) for threshold in thresholds ]
        
        print("thresholds",thresholds)
        print("numsofpeaks",numsofpeaks)

        thresholds_good = thresholds[np.where(np.array(numsofpeaks) == 7)[0]]
        threshold_goodmean = np.mean(thresholds_good)
        
        print("suitable thresholds:", thresholds_good, ". Use: ",threshold_goodmean)

        return(self.spatial_autocorrelation_field_detection(threshold_goodmean, neighborhood_size))
    
    
    
    def calculate_doughnut(self, threshold = 0.1, neighborhood_size = 5):
        
        self.spatial_autocorrelation_field_detection(threshold = threshold, neighborhood_size = neighborhood_size)
            
        # get fields
        x,y = self.spatial_autocorrelation_field

        maxradius = np.min(np.array(self.spatial_autocorrelation_map.shape))/2

        # get midpoint
        midpoint = np.array(self.spatial_autocorrelation_map.shape)/2
        
        # find proper dimensions for doughnut
        self.points_inside_dougnut = []
        r_outer_range = np.linspace(0,maxradius,100)
        r_outer_radii = []
        for r_outer in r_outer_range:
            points_inside_dougnut= [ (x_,y_) for x_,y_ in zip(x,y) if math.dist(midpoint, [x_,y_]) < r_outer ]
            if(len(points_inside_dougnut)>=7):
                if not len(r_outer_radii):
                    self.points_inside_dougnut = points_inside_dougnut
                r_outer_radii.append(r_outer)

        if len(r_outer_radii):

            r_outer_radius_contains6 = r_outer_radii[0] # np.mean(r_outer_radii)

            r_outer_radius_use = r_outer_radius_contains6*1.3
            r_inner_radius_use = r_outer_radius_contains6*0.5

            r_outer_radius_use = np.round(np.min([r_outer_radius_use, maxradius*0.9]))

        else:
            r_outer_radius_use = maxradius*0.9
            r_inner_radius_use = r_outer_radius_use/1.3*0.9
            
            
        # use the autocorrelation map to modify doughnut
        doughnut = self.spatial_autocorrelation_map.copy()

        outsidedoughnut = np.array([ np.array([x_,y_]) for x_,y_ in np.ndindex(self.spatial_autocorrelation_map.shape) if math.dist(midpoint, [x_,y_]) < r_inner_radius_use or math.dist(midpoint, [x_,y_]) > r_outer_radius_use ])
        outsidedoughnut = (outsidedoughnut[:,0], outsidedoughnut[:,1])
        doughnut[outsidedoughnut] = np.nan
        
        self.doughnut = doughnut
        self.autocorr_midpoint = midpoint
        self.r_outer_radius_use = r_outer_radius_use
        self.r_inner_radius_use = r_inner_radius_use

    
            
    def correlation_from_doughnut_rotation(self, degree):
        
        """
        Method of the Spatial_properties class to calculate the correlations for different angles of rotation of the doughnut. 
        Return
        correlation
        """
        
        if not hasattr(self, 'doughnut'):
            raise TypeError('You need to call spatial_properties.calculate_doughnut() before calling this function')
        
        # get the center of the image    
        (h, w) = self.doughnut.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # rotate by degree°, same scale
        M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
        self.doughnut_rotated = cv2.warpAffine(self.doughnut, M, (w, h), borderValue = np.nan)    
    
        indices = np.logical_and(~np.isnan(self.doughnut), ~np.isnan(self.doughnut_rotated))
        
        if np.sum(indices) <= 2:
            return np.nan
        
        r,p = pearsonr(self.doughnut[indices],self.doughnut_rotated[indices])
    
        return r
    
    
    def grid_score(self, threshold=0.1, neighborhood_size=5):
        
        """
        Method of the Spatial_properties class to calculate the grid score.
        Return
        grid score 
        """
        self.calculate_doughnut(threshold = threshold, neighborhood_size = neighborhood_size)

        rotations60 = [60, 120]
        rotations30= [30, 90, 150]

        corr60 = [self.correlation_from_doughnut_rotation(degree) for degree in rotations60]
        corr30 = [self.correlation_from_doughnut_rotation(degree) for degree in rotations30]

        grid_score = np.mean(corr60)-np.mean(corr30)

        return grid_score
    
    
    def shuffle_grid_score(self, iterations=500, cm_per_bin=2, smoothing_sigma_cm=2, smoothing=True ,percentile=95):
        """
        Get a distribution of grid score that would be expected by chance for this neuron

        Argument:
        iterations: How many shufflings to perform
        cm_per_bin: cm per bin in the firing rate map
        smoothing_sigma_cm: smoothing in the firing rate map
        smoothing: smoothing in the firing rate map
        percentile: percentile of the distribution of shuffled grid scores that is used to get the significance threshold

        Return
        tuple: 
        0: 1D numpy array with the grid scores obtained by chance for this neuron
        1: significance threshold for grid score
        
        Example
        
        # get a neuron and set intervals
        n = cg.neuron_list[7]
        n.spike_train.set_intervals(aSes.intervalDict[cond])
        n.spatial_properties.ap.set_intervals(aSes.intervalDict[cond])

        # get the observed value for information score
        n.spatial_properties.firing_rate_map_2d(cm_per_bin=2, smoothing=True)    
        GS = n.spatial_properties.grid_score()

        # get the shuffled values for grid score
        shuGS,threshold = n.spatial_properties.shuffle_grid_score(iterations=100, cm_per_bin=2,percentile=95)

        # plot the results for this neuron
        res = plt.hist(shuGS,label="shuffled")
        ymax=np.max(res[0])
        plt.plot([threshold,threshold],[0,ymax],c="black",label="Threshold")
        plt.plot([GS,GS],[0,ymax],c="red",label="Observed")
        plt.xlabel("Grid score")
        plt.ylabel("Count")
        plt.legend()
        plt.show()
        """
        
        # keep a copy of the pose that we started with
        pose_at_start = self.ap.pose.copy()
        
        self.grid_shuffle=np.empty(iterations)
        for i in range(iterations):
            self.ap.roll_pose_over_time() # shuffle the position data 
            self.firing_rate_map_2d(cm_per_bin=cm_per_bin, smoothing=smoothing, smoothing_sigma_cm=smoothing_sigma_cm) # calculate a firing rate map
            self.grid_shuffle[i] = self.grid_score() # calculate the grid score from the new map
            self.ap.pose=pose_at_start

        # calculate the threshold
        self.grid_score_threshold =  np.percentile(self.grid_shuffle,percentile)
        
        
        return self.grid_shuffle, self.grid_score_threshold
    
    
    def grid_info(self):
        """
        Method to get additional information about the hexagonal grid of the autocorrelation
        
        Returns: Orientation and Spacing of grid (= rotation, radius of hexagon) , error of closest hexagon found, the rotated hexagon
        False if there was an invalid doughnut (not 6 points found using the field detection)
        """
        
        # print(self.points_inside_dougnut)
        
        if not hasattr(self, 'points_inside_dougnut'):
            raise TypeError('You need to call calculate_doughnut() or grid_score() before calling this function')
    
        if len(self.points_inside_dougnut) != 7:
            return False
    
        # get distance of all points to midpoint
        dists = [ math.dist(self.autocorr_midpoint, point) for point in self.points_inside_dougnut]
        # remove midpoint
        pois,dists = np.transpose(np.array([ [poi,dist] for i,(poi,dist) in enumerate(zip(self.points_inside_dougnut,dists)) if i!=np.argmin(dists) ], dtype=object))
        # print("pois=",pois)
        # calculate median distance and use it as radius for hexagon matching
        radius = np.median(dists)
        # print("radius=",radius)
        self.hexagon_radius = radius
        
        # one hexagon
        # hexagon = np.array([ self.autocorr_midpoint+radius*np.array([np.cos(k*2*np.pi/6),np.sin(k*2*np.pi/6)]) for k in range(6)  ])
        # print(hexagon)
        # ax.scatter(hexagon[:,0],hexagon[:,1] , color="blue")
        
        # multiple rotations (1/6 full rotation max due to rotation symmetry, alternatively: (k+fraction of partial rotation)*2pi/6)
        rotations = np.linspace(0,2*np.pi/6,100)
        # create hexagons
        hexagons_rotated = np.array([ [ self.autocorr_midpoint+radius*np.array([np.cos(k*2*np.pi/6 + alpha), np.sin(k*2*np.pi/6 + alpha)]) for k in range(6) ] for alpha in rotations ])
        # print(hexagons_rotated.shape) # = (100, 6, 2)
        # ax.scatter(hexagons_rotated[:,:,0],hexagons_rotated[:,:,1] , color="blue")  # - plot all hexagons
        
        ## for hexagon, rotation in zip(hexagons_rotated, rotations):
        ##     #print("rotation",rotation)
        ##     dist_sum = np.sum([ np.min([ math.dist(hexagon_poi, doughnut_poi) for doughnut_poi in self.points_inside_dougnut ]) for hexagon_poi in hexagon ])
        ##     #print("dist_sum",dist_sum)
            
        # find distance from hexagon points to doughnut points and find best match
        dist_sums = [ np.sum([ np.min([ math.dist(hexagon_poi, doughnut_poi) for doughnut_poi in self.points_inside_dougnut ]) for hexagon_poi in hexagon ]) for hexagon in hexagons_rotated ] # metric dist(X,Y) = sqrt(dist(x1,x2)**2 + dist(y1,y2)**2)
        dist_sum_min_index = np.argmin(dist_sums)
        dist_sum = dist_sums[dist_sum_min_index]
        # print("best rotation at ",rotations[dist_sum_min_index], "using index",dist_sum_min_index, "with error",dist_sum)
        
        hexagon_rotated = hexagons_rotated[dist_sum_min_index]
        #ax.scatter(hexagon_rotated[:,0], hexagon_rotated[:,1] , color="blue")
        # add first point to the end so that you can plot the closed polygon using ax.plot
        hexagon_rotated_ = np.append(hexagon_rotated,np.array([hexagon_rotated[0]]),axis=0)
        #print(hexagon_rotated_.shape) # = (7, 2)
        #ax.plot(hexagon_rotated_[:,0], hexagon_rotated_[:,1] , color="blue")
        #-#for [from_x,from_y],[to_x,to_y] in zip(hexagon_rotated[:-1],hexagon_rotated[1:]): #ax.plot()
            
        return self.hexagon_radius, rotations[dist_sum_min_index], dist_sum, hexagon_rotated_
        
    
    
    def map_crosscorrelation(self, trial1=None, trial2=None, map1=None, map2=None, cm_per_bin=2, smoothing_sigma_cm=2, smoothing=True, xy_range=None):
        
        """
        Method of the Spatial_properties class to calculate the crosscorrelation between 2 firing rate maps which can be specified by giving the trial numbers or by providing 2 maps. 
        Return
        correlation
        """
        if (trial1==None or trial2==None) and (map1.all()==None or map2.all()==None):
            raise TypeError("You have to specify 2 maps or 2 trials")
            
        if isinstance(trial1,int) and isinstance(trial2,int):
            if len(self.ap.ses.trial_intervals.inter) < trial2 or len(self.ap.ses.trial_intervals.inter) < trial1:
                raise TypeError("The indicated trial does not exist.")
            if trial2 == 0 or trial1 == 0:
                raise TypeError("Trial numbering starts at 1.")

            trial1_inter = self.ap.ses.trial_intervals.inter[(trial1-1):trial1,:]
            trial2_inter = self.ap.ses.trial_intervals.inter[(trial2-1):trial2,:]

            # create firing rate maps:
            self.st.unset_intervals()
            self.ap.unset_intervals()
            self.st.set_intervals(trial1_inter)
            self.ap.set_intervals(trial1_inter)
            self.firing_rate_map_2d(cm_per_bin = cm_per_bin, smoothing_sigma_cm = smoothing_sigma_cm, smoothing=smoothing, xy_range=xy_range)
            map1 = self.firing_rate_map

            self.st.unset_intervals()
            self.ap.unset_intervals()
            self.st.set_intervals(trial2_inter)
            self.ap.set_intervals(trial2_inter)
            self.firing_rate_map_2d(cm_per_bin = cm_per_bin, smoothing_sigma_cm = smoothing_sigma_cm, smoothing=smoothing, xy_range=xy_range)
            map2 = self.firing_rate_map
        
    
        # check for dimensions
        if map1.shape != map2.shape:
            raise TypeError("The firing rate maps have different dimensions ("+str(map1.shape)+" != "+str(map2.shape)+"). You have to specify the xy range.")
            
        # calculate crosscorrelation (for valid indices only, nan might be changed to -1 after autocorrelation was calculated)
        indices = np.logical_and(np.logical_and(~np.isnan(map1), ~np.isnan(map2)) , np.logical_and(map1 != -1, map2 != -1))
        r,p = pearsonr(map1[indices],map2[indices])
    
        return r

    
    def identify_wall(self, number_in_array, counts, walls):
    #the goal is to identify the 2 horizontal or the 2 vertical walls in a square arena
        if len(walls)<2:
            #one wall is where the number of border pixels is highest
            array_index = np.where(counts==np.max(counts))
            #if 2 numbers occur equally often
            if array_index[0].shape[0]>1:
                array_index=array_index[0]
            wall = number_in_array[array_index[0]]
            #remove from the lists so that the second wall can also be found
            number_in_array = np.delete(number_in_array, array_index[0])
            counts = np.delete(counts, array_index[0])
            #if the function has already identified one wall, check if the putative other wall has sufficient distance
            if len(walls)==1:
                if wall in range(int(walls[0]-2),int(walls[0]+2)):
                    #if the 2 walls are too close, find a different second wall
                    self.identify_wall(number_in_array, counts, walls)
                else:
                    #otherwise we are done
                    walls.append(wall)
                    return(walls)
            #if this is the first wall to be identified, call the function again to find the other one
            else:
                walls.append(wall)
                self.identify_wall(number_in_array, counts, walls)
        else:
            return(walls)
    
    
    def border_score(self, xy_range, arena, cm_per_bin=2, smoothing=True, smoothing_sigma_cm=2, min_pixel_number_per_field=15, min_peak_rate=4, min_fraction_of_peak_rate=0.4):
        """
        Calculate the border score of a neuron. 
        Score is calculated like in the first border cell paper: https://www.science.org/doi/suppl/10.1126/science.1166466/suppl_file/solstad.som.pdf

        Arguments:
        see arguments of spatial_properties.firing_rate_map_2d()
        The xy-range needs to be specified.
        see arguments of spatial_properties.firing_rate_map_field_detection()
        The default values might still have to be adjusted to optimize field detection.

        Return
        border score
    
        """
        
        if not arena in ["square","circle"]:
            print("Unsupported arena shape; arena needs to be square or circle")

        # For border detection, the arena borders must not touch the borders of the occupancy map. Set the xy-range accordingly when creating the firing rate map.
        self.firing_rate_map_2d(cm_per_bin=cm_per_bin, smoothing_sigma_cm=smoothing_sigma_cm, smoothing=smoothing, xy_range=xy_range)
        
        # get the firing fields of the cell
        self.firing_rate_map_field_detection(min_pixel_number_per_field=min_pixel_number_per_field, min_peak_rate=min_peak_rate, min_fraction_of_peak_rate=min_fraction_of_peak_rate, cm_per_bin=cm_per_bin)
        field_pixel = self.firing_rate_map_fields.copy()
        field_pixel_array=[(x,y) for i,n in enumerate(field_pixel) for x,y in field_pixel[i]]

        #detect the borders in the occupancy map
        border_map = self.ap.detect_border_pixels_in_occupancy_map()
        border_pixel_array = np.where(border_map!=0)

        # get the border pixels in form of an array of tuples (x,y)
        border_pixel = [(np.asarray(x),np.asarray(y)) for x,y in zip(border_pixel_array[0], border_pixel_array[1])]

        # initialize variables
        number_common_pixels = np.zeros(len(field_pixel))
        distance_to_border=np.zeros(border_map.shape[0]*border_map.shape[1])
        distance_to_border=np.reshape(distance_to_border, (border_map.shape[0],border_map.shape[1]))
        border_map_indices=[(np.asarray(x),np.asarray(y)) for (x,y) in np.nditer(np.meshgrid(range(border_map.shape[0]),range(border_map.shape[1])))]
        if arena=="square":
            best_wall=np.zeros(len(field_pixel))

        #no need to run analysis if there are no fields
        if field_pixel:
            #to calculate CM for a rectangular arena, we need to identify the 4 borders separately
            if arena=="square":
                #get horizontal walls
                number_in_array, counts = np.unique(border_pixel_array[0], return_counts=True)
                horizontal_walls=[]    
                self.identify_wall(number_in_array,counts,horizontal_walls)
                #get vertical walls
                number_in_array, counts = np.unique(border_pixel_array[1], return_counts=True)
                vertical_walls=[]    
                self.identify_wall(number_in_array,counts,vertical_walls)

                horizontal_walls=np.sort(horizontal_walls, axis=None)
                vertical_walls=np.sort(vertical_walls, axis=None)

                #assign all border pixels to one of the 4 walls
                #get the wall pixels
                h1_wall_indices=[(np.asarray(x),np.asarray(y)) for x,y in zip(np.repeat(horizontal_walls[0],np.abs(vertical_walls[1]-vertical_walls[0])+1), range(int(vertical_walls[0]),int(vertical_walls[1]+1)))]
                h2_wall_indices=[(np.asarray(x),np.asarray(y)) for x,y in zip(np.repeat(horizontal_walls[1],np.abs(vertical_walls[1]-vertical_walls[0])+1), range(int(vertical_walls[0]),int(vertical_walls[1]+1)))]
                v1_wall_indices=[(np.asarray(x),np.asarray(y)) for x,y in zip(range(int(horizontal_walls[0]),int(horizontal_walls[1]+1)), np.repeat(vertical_walls[0],np.abs(horizontal_walls[1]-horizontal_walls[0])+1))]
                v2_wall_indices=[(np.asarray(x),np.asarray(y)) for x,y in zip(range(int(horizontal_walls[0]),int(horizontal_walls[1]+1)), np.repeat(vertical_walls[1],np.abs(horizontal_walls[1]-horizontal_walls[0])+1))]

                wall_h1=[]; wall_h2=[]; wall_v1=[]; wall_v2=[]
                walls=[wall_h1, wall_h2, wall_v1, wall_v2]

                for b,b_pixel in enumerate(border_pixel):
                    #calculate the distance of each border pixel to the 4 walls
                    distance=[np.sqrt((pixel[0]-b_pixel[0])**2+(pixel[1]-b_pixel[1])**2) for pixel in h1_wall_indices]
                    distance_to_h1=np.nanmin(distance)
                    distance=[np.sqrt((pixel[0]-b_pixel[0])**2+(pixel[1]-b_pixel[1])**2) for pixel in h2_wall_indices]
                    distance_to_h2=np.nanmin(distance)
                    distance=[np.sqrt((pixel[0]-b_pixel[0])**2+(pixel[1]-b_pixel[1])**2) for pixel in v1_wall_indices]
                    distance_to_v1=np.nanmin(distance)
                    distance=[np.sqrt((pixel[0]-b_pixel[0])**2+(pixel[1]-b_pixel[1])**2) for pixel in v2_wall_indices]
                    distance_to_v2=np.nanmin(distance)

                    wall_distances=[distance_to_h1, distance_to_h2, distance_to_v1, distance_to_v2]
                    wall_index=np.where(wall_distances==np.nanmin(wall_distances))
                    if len(wall_index[0])>1:
                        wall_index=wall_index[0][0]
                    else:
                        wall_index=wall_index[0]
                    walls[int(wall_index)].append(b_pixel)
   
            #calculate the distance of all map pixels to the closest border
            for p,pixel in enumerate(border_map_indices):
                distance=[np.sqrt((pixel[0]-b_pixel[0])**2+(pixel[1]-b_pixel[1])**2) for b_pixel in border_pixel]
                distance_to_border[pixel]=np.nanmin(distance)

            for number_of_fields,field in enumerate(field_pixel):
                if arena=="circle":
                    # to find the field that shares most pixels with the borders, get the number of common field-border pixels for the field
                    common_pixels=[b for b,f in itertools.product(border_pixel,field) if b==f]
                    number_common_pixels[number_of_fields]=len(common_pixels)
                else:
                    number_common_pixels_per_wall=np.zeros(len(walls))
                    for m,wall in enumerate(walls):
                        # to find the field that shares most pixels with one border, get the number of common field-border pixels for the field for each border
                        common_pixels=[b for b,f in itertools.product(wall,field) if b==f]
                        number_common_pixels_per_wall[m]=len(common_pixels)
                    #select the highest number
                    highest_number_common_pixels_per_wall=np.nanmax(number_common_pixels_per_wall)
                    best_wall_index=np.where(number_common_pixels_per_wall==highest_number_common_pixels_per_wall)[0][0]
                    if type(highest_number_common_pixels_per_wall)=="list":
                        highest_number_common_pixels_per_wall=highest_number_common_pixels_per_wall[0]

                    number_common_pixels[number_of_fields]=highest_number_common_pixels_per_wall
                    best_wall[number_of_fields]=len(walls[best_wall_index])


            # DM calculation
            DM=np.sum([distance_to_border[x,y]*self.firing_rate_map[x,y] for x,y in field_pixel_array])/len(field_pixel_array)
            #normalize by dividing by the highest mean firing rate in a field pixel
            DM=DM/np.nanmax([self.firing_rate_map[x,y] for (x,y) in field_pixel_array])
            # normalize by dividing by the max distance of a pixel in the map to a border
            DM=DM/np.nanmax(distance_to_border)

            # CM is the maximal number of pixels common between border pixels and field pixel normalized by the number of border pixels
            CM=np.nanmax(number_common_pixels)
            if type(CM)=="list":
                CM=CM[0]
            if arena=="circle":
                #border cells usually don't take up more than half of the border
                CM=CM/(len(border_pixel)/2)
            else:
                #get the wall which is most covered by a field
                best_field_wall_index=np.where(number_common_pixels==CM)[0][0]
                CM=CM/best_wall[best_field_wall_index]

            border_score=(CM-DM)/(CM+DM)
        else:
            border_score=np.nan
            
        return border_score
    
    
    def shuffle_border_score(self, xy_range, arena, iterations=500, cm_per_bin=2, smoothing_sigma_cm=2, smoothing=True, percentile=95):
        """
        Get a distribution of border score that would be expected by chance for this neuron

        Argument:
        iterations: How many shufflings to perform
        cm_per_bin: cm per bin in the firing rate map
        smoothing_sigma_cm: smoothing in the firing rate map
        smoothing: smoothing in the firing rate map
        percentile: percentile of the distribution of shuffled border scores that is used to get the significance threshold

        Return
        tuple: 
        0: 1D numpy array with the border scores obtained by chance for this neuron
        1: significance threshold for border score
        
        Example
        
        # get a neuron and set intervals
        n = cg.neuron_list[7]
        n.set_spatial_properties(ap)
        n.spike_train.set_intervals(aSes.intervalDict[cond])
        n.spatial_properties.ap.set_intervals(aSes.intervalDict[cond])

        # get the observed value for border score
        # it is important to set the xy-range for the occupancy map so that there is some space between the arena borders and the map borders
        # because otherwise the border detection will not work properly
        BS = n.spatial_properties.border_score(xy_range=np.array([[0,0],[125,125]]))

        # get the shuffled values for border score
        shuBS,threshold = n.spatial_properties.shuffle_border_score(iterations=100, cm_per_bin=2,percentile=95, xy_range=np.array([[0,0],[125,125]]))

        # plot the results for this neuron
        res = plt.hist(shuBS,label="shuffled")
        ymax=np.max(res[0])
        plt.plot([threshold,threshold],[0,ymax],c="black",label="Threshold")
        plt.plot([BS,BS],[0,ymax],c="red",label="Observed")
        plt.xlabel("Border score")
        plt.ylabel("Count")
        plt.legend()
        plt.show()
        """
        
        # keep a copy of the pose that we started with
        pose_at_start = self.ap.pose.copy()
        
        self.border_shuffle=np.empty(iterations)
        for i in range(iterations):
            self.ap.roll_pose_over_time() # shuffle the position data 
            # no need to recalculate the firing rate map as it will be calculated in course of border score calculation
            self.border_shuffle[i] = self.border_score(cm_per_bin=cm_per_bin, smoothing=smoothing, smoothing_sigma_cm=smoothing_sigma_cm, xy_range=xy_range, arena=arena) # calculate the border score from the new map
            self.ap.pose=pose_at_start

        # calculate the threshold
        if not np.isnan(self.border_shuffle).all():
            shuffled = self.border_shuffle[~np.isnan(self.border_shuffle)]
        else:
            shuffled = np.nan

            
        self.border_score_threshold =  np.percentile(shuffled,percentile)
        
        
        return self.border_shuffle, self.border_score_threshold
    
    
    def set_intervals(self, inter=None):
        """
        Set the interval for both the Neuron spike train, and the Animal pose
        Argument: The interval (if not provided, only reset)
        Returns nothing, but the intervals are set
        """
        # reset and set interval
        self.st.unset_intervals()
        self.ap.unset_intervals()
        if inter is not None:
            self.st.set_intervals(inter)
            self.ap.set_intervals(inter)
        
        ## clear intervals
        # n.spike_train.unset_intervals()
        # ap.unset_intervals()
        ## set to entire session
        # n.spike_train.set_intervals(ses.trial_intervals.inter)
        # ap.set_intervals(ses.trial_intervals.inter)
    
        
