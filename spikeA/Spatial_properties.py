
import numpy as np
from spikeA.Animal_pose import Animal_pose
from spikeA.Spike_train import Spike_train
import spikeA.spatial_properties # this has no capital letter, so refers to the c code
import spikeA.animal_pose # this has no capital letter, so refers to the c code
from scipy.interpolate import interp1d
from scipy import ndimage
from scipy.ndimage import sum as ndi_sum
from scipy.ndimage import center_of_mass as ndi_center_of_mass
from skimage.measure import EllipseModel
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
        Method to calculate the position of each spike of the Spike_train object in 2D.
        
        The spike position is not stored in the Spatial_properties object but returned by the function
        This behavior was implement to save memory when working thousands of Spatial_properties objects in memory.
               
        
        Return
        2D numpy array, 2 columns with x and y position associated with each spike
        """
        
        ## check that the Animal_pose and Spike_train object have the same intervals
        ## this check is required because if there are spikes at time points with invalid pose values, they will be interpolated to a valid value.
        if self.ap.intervals.inter.shape != self.st.intervals.inter.shape:
            raise ValueError("The intervals in the Animal_pose and Spike_train objects are not the same. Please make sure the intervals in the Animal_pose and Spike_train objects are the same before calling spike_position()")
        if np.any(self.ap.intervals.inter != self.st.intervals.inter):
            raise ValueError("The intervals in the Animal_pose and Spike_train objects are not the same. Please make sure the intervals in the Animal_pose and Spike_train objects are the same before calling spike_position()")
        
        fx = interp1d(self.ap.pose[:,0], self.ap.pose[:,1], bounds_error=False)
        fy = interp1d(self.ap.pose[:,0], self.ap.pose[:,2], bounds_error=False)

        # create a numpy 2D array to store the spike position
        spike_posi = np.empty((self.st.st.shape[0],2))
        
        # get the position of the animal at each spike time
        spike_posi[:,0] = fx(self.st.st)
        spike_posi[:,1] = fy(self.st.st)
        
        self.spike_posi = spike_posi
        
        return spike_posi
    
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
        
        # get radians from sin and cos
        self.spike_hd = np.arctan2(ihds,ihdc)
        
        return self.spike_hd
        
        
        
    def firing_rate_head_direction_histogram(self, deg_per_bin=10, smoothing_sigma_deg=10, smoothing=True, recalculate_hd_occupancy_histo=True):
        """
        Method of the Spatial_properties class to calculate the firing rate of a neuron as a function of head direction.
        
        If a compatible occupancy_hd histogram is not already present in the self.animal_pose object, one will be calculated.
        
        Calculations are all done in radians.
        
        Arguments:
        deg_per_bin: degrees per bins in the head-direction histogram
        smoothing_sigma_deg: standard deviation of the gaussian kernel used to smooth the firing rate head direction histogram
        smoothing: boolean indicating whether or not smoothing is applied
        recalculate_hd_occupancy_histo: force to call ap.head_direction_occupancy_histogram to generate a new hd occupancy
        
        Return:
        The Spatial_properties.firing_rate_head_direction_histo is set. It is a 1D numpy array containing the firing rate in Hz as a function of head direction.
        """
        self.hd_histo_deg_per_bin = deg_per_bin
        self.hd_histo_smoothing_sigma_deg = smoothing_sigma_deg
        self.hd_histo_smoothing = smoothing
        
      
        # create a new hd occupancy histogram (if needed or desired)        
        if not hasattr(self.ap, 'hd_occupancy_histogram') or recalculate_hd_occupancy_histo:
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
        
        """
        Method to calculate firing rate of HD cells like using firing_rate_head_direction_histogram()
        with binned intervals defined in sub_intervals (might use times2intervals() function)

        Arguments:
            sub_intervals: (2,n) array that contains the intervals

        Returns: hd firing, mean vector length, mean direction on each interval
        """

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
            
        
    def firing_rate_map_2d(self,cm_per_bin=2, smoothing_sigma_cm=2, smoothing = True, xy_range=None, recalculate_occupancy_map = True,remove_spike_in_occupancy_gaps = False, minimal_occupancy_sec=0.0):
        """
        Method of the Spatial_properties class to calculate a firing rate map of a single neuron.
        
        If a compatible occupancy map is not already present in the self.animal_pose object, one will be calculated.
        
        Arguments
        cm_per_bin: cm per bins in the firing rate map
        smoothing_sigma_cm: standard deviation of the gaussian kernel used to smooth the firing rate map
        smoothing: boolean indicating whether or not smoothing should be applied to the firing rate map
        xy_range: 2D np.array of size 2x2 [[xmin,ymin],[xmax,ymax]] with the minimal and maximal x and y values that should be in the occupancy map. This can be used to set the firing rate map to a specific size. The default value is None, which means that the size of the occupancy map (and firing rate map) will be determined from the range of values in the Animal_pose object.
        recalculate_occupancy_map: boolean indicating whether the occpuancy map should be recalculated
        remove_spike_in_occupancy_gaps: boolean indicating whether we should remove spikes from 2D histogram spike count when they fall into a non-visited location in the occupancy map
        
        Return
        The Spatial_properties.firing_rate_map is set. It is a 2D numpy array containing the firing rate in Hz in a set of bins covering the environment.
        """
        # we could do check for valid value ranges
        self.map_cm_per_bin = cm_per_bin
        self.map_smoothing_sigma_cm = smoothing_sigma_cm
        self.map_smoothing = smoothing
        
        # create a new occupancy map (if needed or desired)
        if not hasattr(self.ap, 'occupancy_map_2d') or recalculate_occupancy_map:
            self.ap.occupancy_map_2d(cm_per_bin =self.map_cm_per_bin, 
                                     smoothing_sigma_cm = self.map_smoothing_sigma_cm, 
                                     smoothing = smoothing, zero_to_nan = True,xy_range=xy_range)
        
        ## get the position of every spike
        spike_posi = self.spike_position()
        
        ## calculate the number of spikes per bin in the map
        ## we use the bins of the occupancy map to make sure that the spike count maps and occupancy map have the same dimension
        spike_count,x_edges,y_edges = np.histogram2d(x = spike_posi[:,0], 
                                                     y= spike_posi[:,1],
                                                     bins= self.ap.occupancy_bins)
        
        # remove spikes that fell into a bin that the animal was not seen in.
        # usually not required but was needed when calculating control firing rate maps.
        if remove_spike_in_occupancy_gaps: 
            spike_count[np.isnan(self.ap.occupancy_map)] = 0
            
        # remove spikes with insufficient occupancy duration 
        spike_count[self.ap.occupancy_map < minimal_occupancy_sec] = 0
        #spike_count[self.ap.occupancy_map < minimal_occupancy_sec] = np.nan
        
        
        # save this for later (e.g., in information_score())
        self.spike_count = spike_count
        
        ## smooth the spike count array
        if smoothing:
            spike_count = ndimage.filters.gaussian_filter(spike_count,
                                                          sigma=self.map_smoothing_sigma_cm/self.map_cm_per_bin)
    
        ## get the firing rate in Hz (spike count/ time in sec)
        self.firing_rate_map = spike_count/self.ap.occupancy_map
    
        self.map_xy_range = np.array([[self.ap.occupancy_bins[0][0],self.ap.occupancy_bins[1][0]],
                                      [self.ap.occupancy_bins[0][-1],self.ap.occupancy_bins[1][-1]]])
    
    def firing_rate_map_peak_location(self):
        """
        Function to get the location of the peak rate within the firing rate map
        
        You need to call firing_rate_map_2d() before running this function.
        
        Return
        Tuple with 2 elements: a numpy array with x and y indices of peak in the map, a numpy array with the x and y location of the peak in cm coordinates.
        """
        if not hasattr(self, 'firing_rate_map'):
                raise ValueError('Call self.firing_rate_map_2d() before calling self.firing_rate_map_peak_location()')
    
        myMap = self.firing_rate_map.copy()
        myMap[np.isnan(myMap)]=-1.0
        xy = np.array(np.unravel_index(np.argmax(myMap), myMap.shape))
        grid_peak_location = self.map_xy_range[0] + xy * self.map_cm_per_bin
        return xy, grid_peak_location
    
    
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
        spike_posi = self.spike_position() 
        
        spike_count,x_edges = np.histogram(spike_posi[:,0],bins= self.ap.occupancy_bins)
        
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
        
        Usage 
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
        
        
   
    
    def spatial_autocorrelation_map_2d(self,min_n_for_correlation=10,invalid_to_nan=False):
        """
        Method of the Spatial_properties class to calculate a spatial autocorrelation map of a single neuron.
        
        If a compatible firing rate map is not already present in the spatial_properties object, an error will be given.
        
        The values in the autocorrelation ranges from -1 to 1. They are Pearson correlation coefficient between firing rate values at a given spatial offset
        
        Arguments
        min_n_for_correlation: Minimal number of paired firing rates at a given offset that is needed to calculate a correlation coefficient. Adjust to prevent high variability towards the edges to the autocorrelation map.
        invalid_to_nan: Set the invalid data point in the spatial autocorrelation to np.nan instead of -2.0. The c function returns invalid values as -2.0.
        
        Return
        The Spatial_properties.spatial_autocorrelation_map is set. It is a 2D numpy array.
        """
        
        ## check for firing rate map
        if not hasattr(self, 'firing_rate_map'):
            raise TypeError("Call spatial_properties.firing_rate_map_2d() before calling spatial_properties.spatial_autocorrelation_map_2d()")
        
        ## check for smoothing
        if not self.map_smoothing:
            print("You should smooth the firing rate map when calculating autocorrelation in order to detect the fields")
        
        
        frm = self.firing_rate_map.copy() # to avoid changing this map, as the user would not be aware of this change
        
        ## convert nan values to -1 for C function
        frm[np.isnan(frm)]=-1.0
        
        ## create an empty array of the appropriate dimensions to store the autocorrelation data
        ## size of autocorrelation is 2*map + 1, always an odd number.
        auto_array = np.zeros((2*frm.shape[0]+1,2*frm.shape[1]+1))

        ## create the spatial autocorrelation calling a C function
        spikeA.spatial_properties.map_autocorrelation_func(frm,auto_array,min_n_for_correlation)
        
        if invalid_to_nan:
            auto_array[auto_array==-2.0]=np.nan
        self.spatial_autocorrelation_map = auto_array

    
    def spatial_crosscorrelation_map_2d(self, firing_rate_map1, firing_rate_map2, min_n_for_correlation=10, valid_radius_cm=None, cm_per_bin = None):
        """
        Method of the Spatial_properties class to calculate a spatial crosscorrelation between two firing rate maps of the same dimensions.
        
        This finds the offsets in the x and y axis at which the firing rate values of map2 are correlated with map1.
        
        
        Arguments:
        firing_rate_map1: 2D Numpy array containing a firing rate map
        firing_rate_map2: 2D Numpy array containing a firing rate map
        min_n_for_correlation: Minimal number of paired firing rates at a given offset that is needed to calculate a correlation coefficient. Adjust to prevent high variability towards the edges to the crosscorrelation map.
        valid_radius: If set, only bins within the valid_radius from the center of the crosscorrelation map will be kept and the rest set to np.nan. Useful if you only want to keep the center of the crosscorrelation map.
        cm_per_bin: If you set valid_radius, you need to specify the cm_per_bin in the maps.
        
        Return
        The spatial crosscorrelation of the 2 firing rate maps. Invalid values are set to np.nan
        """
        
        if firing_rate_map1.ndim != firing_rate_map2.ndim:
            raise ValueError("The dimensions of firing_rate_map1 is the the same as those of firing_rate_map2")
            
        if not np.array_equal(firing_rate_map1.shape,firing_rate_map2.shape):
            raise ValueError("The shape of firing_rate_map1 is the the same as that of firing_rate_map2")
        
        frm1 = firing_rate_map1.copy() # to avoid modifying the input maps
        frm2 = firing_rate_map2.copy()
        
        ## convert nan values to -1.0 for the C function
        frm1[np.isnan(frm1)]=-1.0
        frm2[np.isnan(frm2)]=-1.0
        
        ## create an empty array of the appropriate dimensions to store the autocorrelation data
        cross_array = np.zeros((2*frm1.shape[0]+1,2*frm1.shape[1]+1))
        
        ## call the c function
        spikeA.spatial_properties.map_crosscorrelation_func(frm1,frm2,cross_array,min_n_for_correlation)
        
        
        cross_array[cross_array==-2.0]=np.nan
        
        if valid_radius_cm is not None:
            if cm_per_bin is None:
                raise ValueError("set cm_per_bin when calling spatial_crosscorrelation_map_2d() if using the argument valid_radius_cm")
            
            xs,ys = np.meshgrid(np.arange(0,cross_array.shape[0]),np.arange(0,cross_array.shape[1]))
            midPoint=(cross_array.shape[0]/2,cross_array.shape[1]/2)
            distance = np.sqrt((xs.T-midPoint[0])**2 + (ys.T-midPoint[1])**2) * cm_per_bin
            
            cross_array[distance>valid_radius_cm]=np.nan

        return cross_array

        
    def spike_triggered_short_term_cross_firing_rate_map(self, spatial_properties_neuron_2, cm_per_bin=2, time_window_sec=2, xy_range = None, smoothing_sigma_cm=2, smoothing=True, valid_radius_cm=None):
        """
        Method of the Spatial_properties class that calculate a spike-triggered short-term cross-firing rate map.
        
        This involves two the Spatial_properties objects of 2 neurons. 
        We calculate the a firing rate map for the spikes of neuron2 relative to the spikes of neuron1.
        For each spike of neuron1, we look in a short time period after the spike, where the spikes of neuron2 occurred relative the the position of the trigger spike of neuron1.
        
        The c code assumes that the pose data and the spike times are chronologically sorted. This assumption allowed me to speed up the code.
        
        
        Arguments:
        spatial_properties_neuron_2: spatial properties of a different neuron 
        cm_per_bin: bin size for the map
        time_window_sec: size of the time window following the trigger spike that will be included in the map.
        xy_range: 2D np.array of size 2x2 [[xmin,ymin],[xmax,ymax]] with the minimal and maximal x and y values that should be in the occupancy map. This is used to set the size of the firing rate map. The default value is None, which means that the size of the occupancy map (and firing rate map) will be determined from the range of values in the Animal_pose object (will be double a normal firing rate maps)
        smoothing_sigma_cm: size of smoothing kernel
        smoothing: boolean determining whether smoothing is applied.
        valid_radius: If set, only bins within the valid_radius from the center of the crosscorrelation map will be kept and the rest set to np.nan. Useful if you only want to keep the center of the crosscorrelation map.
        
        Returns:
        2D np.array with the spike-triggered short-time cross-firing rate map
        Number of spikes from n1 that were used
        Number of spikes from n2 that were used
        Sum of spikes in the spike count map
        Total time in the occupancy map
        
        """
        
        self.map_cm_per_bin = cm_per_bin
        self.map_smoothing_sigma_cm = smoothing_sigma_cm
        self.map_smoothing = smoothing
        
        
        invalid = np.isnan(self.ap.pose[:,1:3]).any(axis=1)
        val = self.ap.pose[~invalid,1:3]

        if xy_range is None:
                xy_max = np.ceil(val.max(axis=0))+self.map_cm_per_bin
                xy_min = np.floor(val.min(axis=0))-self.map_cm_per_bin
        else :
            xy_max= np.array(xy_range[1,:])
            xy_min= np.array(xy_range[0,:])
        #print("min and max x and y for the np.arange function : {}, {}".format(xy_min,xy_max))
        
        # We need odd number size to have the center at the center of the map
        # for example a 3x3 map in which bin indexed 1,1 is the center
        occupancy_range_bins = np.ceil((xy_max - xy_min)*2/cm_per_bin).astype(int)
        #print("occupancy_range_bins:", occupancy_range_bins)
        
        occupancy_range_bins[occupancy_range_bins%2==0]+=1 # make sure it is odd number of bins in each dimension
        
        #print("occupancy_range_bins:", occupancy_range_bins)
        occ = np.zeros((int(occupancy_range_bins[0]),int(occupancy_range_bins[1])))
        
        
        spike_posi_n1 = self.spike_position() 
        spike_posi_n2 = spatial_properties_neuron_2.spike_position() 
        spike_used_n2 = np.zeros_like(spike_posi_n2[:,0])
              
        # create an occupancy map of the mouse around the spikes of self
        spikeA.animal_pose.spike_triggered_occupancy_map_2d_func(self.st.st,
                                                                 spike_posi_n1[:,0].copy(order="C"),
                                                                 spike_posi_n1[:,1].copy(order="C"),
                                                                 self.ap.pose[:,0].copy(order="C"),
                                                                 self.ap.pose[:,1].copy(order="C"),
                                                                 self.ap.pose[:,2].copy(order="C"),
                                                                 time_window_sec,
                                                                 occ,
                                                                 cm_per_bin,
                                                                 valid_radius_cm)

        # create a spike count map around the spikes of self
        spike_map = np.zeros((int(occupancy_range_bins[0]),int(occupancy_range_bins[1])))
        spikeA.spatial_properties.spike_triggered_spike_count_2d_func(self.st.st,
                                                                      spike_posi_n1[:,0].copy(order="C"),
                                                                      spike_posi_n1[:,1].copy(order="C"),
                                                                      spatial_properties_neuron_2.st.st,
                                                                      spike_posi_n2[:,0].copy(order="C"),
                                                                      spike_posi_n2[:,1].copy(order="C"),
                                                                      spike_used_n2,
                                                                      time_window_sec,
                                                                      spike_map,
                                                                      cm_per_bin,
                                                                      valid_radius_cm)
        
        # calculate how many spikes of neuron 2 were used in the calculation
        spike_used_n2 = spike_used_n2.sum()
        # calculate how many spikes of neuron 1 were used in the calculation
        spike_used_n1 = spike_posi_n1.shape[0]
        # number of spikes in the spike count map
        spike_count = np.nansum(spike_map)
        # calculate the time in the occupancy map
        occupancy_time = np.nansum(occ)
        
        occs = occ.copy()
        if self.map_smoothing:
            occs = ndimage.filters.gaussian_filter(occs,sigma=self.map_smoothing_sigma_cm/self.map_cm_per_bin)
            spike_map = ndimage.filters.gaussian_filter(spike_map,sigma=self.map_smoothing_sigma_cm/self.map_cm_per_bin)
        
        occs[occ==0.0] = np.nan
        rate_map = spike_map/occs
        
        ## this should be implemented in the c functions so that data past the radius can't influence within the radius via smoothing.
        if valid_radius_cm is not None:
            
            xs,ys = np.meshgrid(np.arange(0,rate_map.shape[0]),np.arange(0,rate_map.shape[1]))
            midPoint=(int(rate_map.shape[0]/2),int(rate_map.shape[1]/2))
            distance = np.sqrt((xs.T-midPoint[0])**2 + (ys.T-midPoint[1])**2) * cm_per_bin
            
            rate_map[distance>valid_radius_cm]=np.nan
        
        return rate_map, spike_used_n1, spike_used_n2, spike_count, occupancy_time
    
        
    def spatial_autocorrelation_field_detection(self, threshold = 0.1, neighborhood_size = 5):
        """
        Method to detect fields based in the spatial autocorrelation map
        
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
        for dx,dy in slices:
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
        """
        This function will call spatial_autocorrelation_field_detection()
        Find the max radius and location of the fields in the spatial autocorrelation
        Determine the doughnut dimensions from the fields
        
        Returns nothing
        It sets self.doughnut, self.autocorr_midpoint, self.r_outer_radius_use, self.r_inner_radius_use
        
        """
        
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

        outsidedoughnut = np.array([ np.array([x_,y_]) for x_,y_ in np.ndindex(self.spatial_autocorrelation_map.shape) if math.dist([midpoint[1],midpoint[0]], [x_,y_]) < r_inner_radius_use or math.dist([midpoint[1],midpoint[0]], [x_,y_]) > r_outer_radius_use ])
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
        
        if np.sum(indices) <= 2: # not enough to calculate a correlation
            return np.nan
        
        if np.std(self.doughnut[indices])==0: # if there is no variability in the data, correlation not possible
            return np.nan
        
        r,p = pearsonr(self.doughnut[indices],self.doughnut_rotated[indices])
    
        return r
    
    def calculate_ellipse_parameters(self):
        """
        An ellipse is fitted to the fields found by calculate_doughnut in the autocorrelation map. Ellipse parameters (eccentricity, ellipticity, ellipse axes and rotation) are set.
        """
        if not hasattr(self, 'doughnut'):
            raise ValueError('Call self.calculate_doughnut() before calling this function')
            
        p_dist=[math.dist(self.autocorr_midpoint, p) for p in self.points_inside_dougnut]
        X=[i[0] for i in self.points_inside_dougnut]
        Y=[i[1] for i in self.points_inside_dougnut]
        #remove middle point
        X=np.delete(X,np.where(p_dist==np.min(p_dist))[0])
        Y=np.delete(Y,np.where(p_dist==np.min(p_dist))[0])
        #fit ellipse to points
        points=np.column_stack([X,Y])
        ellipse = EllipseModel()
        if ellipse.estimate(points):
            xc, yc, a, b, theta = ellipse.params
            #calculate
            self.eccentricity=np.sqrt(1-((np.min([a,b])*np.min([a,b]))/(np.max([a,b])*np.max([a,b]))))
            self.ellipticity=(np.max([a,b])-np.min([a,b]))/np.max([a,b])
            self.ellipse_axes=(a,b)
            self.ellipse_rotation=theta
        else:
            self.ellipticity=np.nan
            self.eccentricity=np.nan
            self.ellipse_axes=(np.nan,np.nan)
            self.ellipse_rotation=np.nan 
            
            
    def correct_pose_ellipticity(self):
        """
        The pose data is rotated and the y axis is stretched/squeezed to make the ellipse into a circle.
        """
        #rotate position data
        xy = np.column_stack([self.ap.pose[:,1],self.ap.pose[:,2]])
        angle = -self.ellipse_rotation #in radian
        mat = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]) #2D rotation matrix
        self.ap.pose[:,1:3] = xy@mat
        self.ap.pose[:,2]=self.ap.pose[:,2]*self.ellipse_axes[1]/self.ellipse_axes[0] #y stretched or sqeezed to match x
        
            
    def grid_score(self, threshold=0.1, neighborhood_size=5, calculate_ellipticity=False, correct_ellipticity=False, keep_corrected_map=False):
        
        """
        Method of the Spatial_properties class to calculate the grid score.
        
        Before running this function, you need to create a firing rate map 2d
        using  Spatial_properties.firing_rate_map_2d()  
        
        If calculate_ellipticity=True, an ellipse is fitted to the fields found by calculate_doughnut in the autocorrelation map. Ellipse parameters (eccentricity and ellipticity) are set.
        
        If correct_ellipticity=True, an ellipse is fitted to the fields found by calculate_doughnut in the autocorrelation map. Ellipse parameters are set. 
        In addition, the pose data is rotated and the y axis is stretched/squeezed to make the ellipse into a circle.
        A new firing rate map, autocorrelation map and doughnut are calculated. Then the position data and the firing rate map are set back to the original versions.
        
        If keep_corrected_map=True, the map created to correct for ellipticity is kept. However, the pose data is still set back. This will only take effect if correct_ellipticity is True.
        
        Return
        grid score 
        """
        
        if not hasattr(self, 'firing_rate_map'):
            raise ValueError('Call self.firing_rate_map_2d() before calling self.grid_score()')
        
        # if the map has a peak firing rate of 0, it is not possible to calculate a grid score
        if np.nanmax(self.firing_rate_map) == 0:
            self.points_inside_dougnut=[]
            if calculate_ellipticity or correct_ellipticity:
                self.ellipticity=np.nan
                self.eccentricity=np.nan
                self.ellipse_axes=(np.nan,np.nan)
                self.ellipse_rotation=np.nan
            return np.nan
        
        self.calculate_doughnut(threshold = threshold, neighborhood_size = neighborhood_size)

        if not self.points_inside_dougnut:
            grid_score=np.nan
            if calculate_ellipticity or correct_ellipticity:
                self.ellipticity=np.nan
                self.eccentricity=np.nan
                self.ellipse_axes=(np.nan,np.nan)
                self.ellipse_rotation=np.nan
            return grid_score
        
        if calculate_ellipticity or correct_ellipticity:
            self.calculate_ellipse_parameters()
        
        #the ellipse should not be stretched/squeezed by more than a factor of 4
        if correct_ellipticity and (self.ellipse_axes[1]/self.ellipse_axes[0] < 4 and self.ellipse_axes[1]/self.ellipse_axes[0] > 0.25):
            #save current ap.pose and firing rate map
            current_pose=self.ap.pose.copy()
            current_firing_rate_map=self.firing_rate_map.copy()

            self.correct_pose_ellipticity()

            #recalculate the firing rate map, autocorrelation (called by calculate_doughnut) and doughnut
            self.firing_rate_map_2d(cm_per_bin=self.map_cm_per_bin, smoothing_sigma_cm = self.map_smoothing_sigma_cm, smoothing=self.map_smoothing)
            self.calculate_doughnut()

            #set back
            self.ap.pose=current_pose
            if not keep_corrected_map:
                self.firing_rate_map=current_firing_rate_map


        rotations60 = [60, 120]
        rotations30= [30, 90, 150]

        
        
        corr60 = [self.correlation_from_doughnut_rotation(degree) for degree in rotations60]
        corr30 = [self.correlation_from_doughnut_rotation(degree) for degree in rotations30]

        grid_score = np.mean(corr60)-np.mean(corr30)
        
      
        
        return grid_score
    
    
    def shuffle_grid_score(self, iterations=500, cm_per_bin=2, smoothing_sigma_cm=2, smoothing=True, percentile=95, correct_ellipticity=False):
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
        
        # keep a copy of the pose and the firing rate map that we started with
        pose_at_start = self.ap.pose.copy()
        current_firing_rate_map=self.firing_rate_map.copy()
        
        
        self.grid_shuffle=np.empty(iterations)
        for i in range(iterations):
            self.ap.roll_pose_over_time() # shuffle the position data 
            self.firing_rate_map_2d(cm_per_bin=cm_per_bin, smoothing=smoothing, smoothing_sigma_cm=smoothing_sigma_cm) # calculate a firing rate map
            self.grid_shuffle[i] = self.grid_score(correct_ellipticity=correct_ellipticity) # calculate the grid score from the new map
        
        #reset the pose data and the firing rate map
        self.ap.pose=pose_at_start
        self.firing_rate_map=current_firing_rate_map
        
        # calculate the threshold
        self.grid_score_threshold =  np.percentile(self.grid_shuffle,percentile)
        
        
        return self.grid_shuffle, self.grid_score_threshold
    
    
    def grid_info(self):
        """
        Method to get additional information about the hexagonal grid of the autocorrelation
        
        Returns: 
        grid spacing in cm, orientation, error of closest hexagon found, the rotated hexagon
        The function returns np.nan values if 6 fields were not detected in the spatial autocorrelation doughnut.
        
        Possible usage:
        grid_info = n.spatial_properties.grid_info()
        if grid_info and np.isfinite(grid_info[0]):
            # valid grid info obtained
        
        """
        
        # print(self.points_inside_dougnut)
        
        if not hasattr(self, 'points_inside_dougnut'):
            raise TypeError('You need to call calculate_doughnut() or grid_score() before calling this function')
    
        if len(self.points_inside_dougnut) != 7:
            return np.nan, np.nan, np.nan, np.nan
    
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
        dist_sums = [ np.sum([ np.min([ math.dist(hexagon_poi, doughnut_poi) for doughnut_poi in self.points_inside_dougnut ])**2 for hexagon_poi in hexagon ]) for hexagon in hexagons_rotated ] # metric dist(X,Y) = sqrt(dist(x1,x2)**2 + dist(y1,y2)**2)
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
            
        return self.hexagon_radius*self.map_cm_per_bin, rotations[dist_sum_min_index], dist_sum, hexagon_rotated_
        
    
    
    def map_pearson_correlation(self, trial1=None, trial2=None, map1=None, map2=None, cm_per_bin=2, smoothing_sigma_cm=2, smoothing=True, xy_range=None):
        
        """
        Method of the Spatial_properties class to calculate a Pearson correlation coefficient between 2 firing rate values of 2 firing rate maps. 
        
        The maps can be passed as parameters or generated within the function.
        
        Arguments:
        trial1:
        trial2:
        map1:
        map2:
        cm_per_bin:
        smoothing_sigma_cm:
        smoothing:
        xy_range:
        
        Return
        Pearson correlation coefficient between two firing rate maps
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

      
    def firing_rate_map_field_detection(self, min_pixel_number_per_field=25, max_fraction_pixel_per_field=0.33, min_peak_rate=4, min_fraction_of_peak_rate=0.45, max_min_peak_rate=10):
        """
        
        Don't use this, use firing_rate_map_field_detection_fast()
        
        
        Method of the Spatial_properties class to calculate the position and size of fields in the firing rate map.
        
        If a compatible firing rate map is not already present in the spatial_properties object, an error will be given.
        Arguments:
        min_pixel_number_per_field: minimal number of pixels so that the putative firing field will be appended to the fields list
        max_fraction_pixel_per_field: maximal field pixels (as fraction of total pixel number)
        min_peak_rate: minimal firing rate in a field
        min_fraction_of_peak_rate: threshold firing rate of a pixel to be considered a field pixel (as fraction of peak rate)
        max_min_peak_rate: maximal threshold firing rate

        Return
        Sets self.firing_rate_map_field_size which is a list with the area of each field
        Sets self.firing_rate_map_fields whi
        """
        ## check for firing rate map
        if not hasattr(self, 'firing_rate_map'):
            raise TypeError("Call spatial_properties.firing_rate_map_2d() before calling spatial_properties.firing_rate_map_field_detection()")
        
        # work on a copy of the firing rate map because fields will be set to nan
        rate_map = self.firing_rate_map.copy()
        
        # invalid pixels should be nan
        rate_map[rate_map==-1.0]=np.nan
        
        # create an empty list to which the detected fields will be appended
        fields = []
        # get the peak rate of the whole map
        peak_rate = np.nanmax(rate_map)
        # call the recursive function detect_one_field which will find all the field
        fields = self.detect_one_field(rate_map, fields, peak_rate, min_pixel_number_per_field, max_fraction_pixel_per_field, min_peak_rate, min_fraction_of_peak_rate, max_min_peak_rate)
        self.firing_rate_map_fields = fields
        # calculate the field size in cm2
        if fields:
            self.firing_rate_map_field_size = [len(fields[i])*self.map_cm_per_bin**2 for i in range(len(fields))]
        else:
            self.firing_rate_map_field_size = []

    def detect_one_field(self, rate_map, fields, peak_rate, min_pixel_number_per_field, max_fraction_pixel_per_field, min_peak_rate, min_fraction_of_peak_rate, max_min_peak_rate):
        """
        
        Don't use this, use firing_rate_map_field_detection_fast()
        
        
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
            self.find_field_pixels(start_pixel, field_pixels, rate_map, peak_rate, min_fraction_of_peak_rate, max_min_peak_rate)
            # set the start pixel to nan so that it will not be selected again as start pixel
            rate_map[start_pixel]=np.nan
            # a firing field must have a minimal number of pixels and should not exceed a certain fraction of the firing rate map
            if len(field_pixels)>min_pixel_number_per_field and len(field_pixels)<(max_fraction_pixel_per_field*rate_map.shape[0]*rate_map.shape[1]):
                fields.append(field_pixels)
                # set all field pixels to nan so that they will not be assigned to other fields
            for p in field_pixels:
                rate_map[p]=np.nan
            # check if there could be more fields (no more fields when all pixels are nan or have too low firing rate)
            if not all(all(np.isnan(rate_map[:,r])) for r in range(rate_map.shape[1])) or all(all(rate_map[:,r]<np.nanmin([peak_rate*min_fraction_of_peak_rate,max_min_peak_rate])) for r in range(rate_map.shape[1])):
                self.detect_one_field(rate_map, fields, peak_rate, min_pixel_number_per_field, max_fraction_pixel_per_field, min_peak_rate, min_fraction_of_peak_rate,max_min_peak_rate)
        return(fields)    
          
    def find_field_pixels(self, p, field_pixels, rate_map, peak_rate, min_fraction_of_peak_rate,max_min_peak_rate):
        """
        
        Don't use this, use firing_rate_map_field_detection_fast()
        
        
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
            if p not in field_pixels and p[0]<rate_map.shape[0] and p[1]<rate_map.shape[1] and rate_map[p]>np.nanmin([peak_rate*min_fraction_of_peak_rate,max_min_peak_rate]):
                field_pixels.append(p)
                # use field pixel as new starting point to detect all pixels belonging to the same firing field
                self.find_field_pixels(p, field_pixels, rate_map, peak_rate, min_fraction_of_peak_rate, max_min_peak_rate)
        

    def firing_rate_map_field_detection_fast(self,min_pixel_number_per_field=20, 
                                             min_peak_rate=4, min_peak_rate_proportion= 0.30):
        """
        Method to detect firing fields in a firing rate map
        
        Run self.firing_rate_map_2d() before calling firing_rate_map_field_detection_fast to get a firing rate map.
        
        It calls a c function that iterates to find the pixels of the field. 
        
        Arguments:
        min_pixels_number_per_fields: minimal number of pixels in a field so that this is a field
        min_peak_rate: minimal peak rate in a firing field
        min_peak_rate_proportion: proportion of the firing rate peak that is required to add a pixel to the field
                
        Returns: 
        A list of dictionary, one dictionary per field. Each dictionary contains the fieldMap, peakRate, rateMap (after detection), fieldPixelCount
        
        Usage:
        
        xy_range=np.array([[-50,-50],[50,50]])
        n.spatial_properties.firing_rate_map_2d(cm_per_bin=3, smoothing_sigma_cm=5, smoothing=True, xy_range=xy_range)
        min_peak_rate = np.nanmax([np.nanmax(n.spatial_properties.firing_rate_map)/2,4])
        fieldList = n.spatial_properties.firing_rate_map_field_detection_fast(min_pixel_number_per_field=10,
                                                                              min_peak_rate= min_peak_rate,
                                                                              min_peak_rate_proportion= 0.4)                                                                     
        for field in fieldList:
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(field["field_map"])
        axes[0].set_title("{:.3f} Hz".format(field["peak_rate"]))
        axes[1].imshow(field["rate_map"])
        axes[1].set_title("{:.3f} Hz".format(np.nanmax(field["rate_map"])))
        plt.show()

        
        
        """
        # check if we have at least one peak in the map
        #if np.nanmax(self.firing_rate_map) < min_peak_rate:
        #    return [],[]
        
        rateMap = self.firing_rate_map.copy() # to avoid destroying the map during field detection, pixels detected in a field will be set to np.nan
        fieldList = []
                
        # in the c code, we will use -1.0 as invalid data instead of np.nan
        rateMap[np.isnan(rateMap)]= -1.0
        
        # loop to attempt to detect fields
        while np.nanmax(rateMap) > min_peak_rate:
            # get a map to show the field
            fieldMap = np.zeros_like(rateMap,dtype=np.int32) # map in which the field pixels are set to 1, we can use integer as it is smaller

            # c function to detect a single field, the pixels of the field in rateMap are set to -1.0 and to 1.0 in fieldMap
            fieldPixelCount = spikeA.spatial_properties.detect_one_field_func(rateMap,fieldMap, min_peak_rate, min_peak_rate_proportion)

            if fieldPixelCount > min_pixel_number_per_field:
                fieldList.append({"field_map": fieldMap,
                                  "peak_rate": np.nanmax(self.firing_rate_map[fieldMap==1]),
                                  "rate_map": rateMap.copy(),
                                  "field_pixel_count": fieldPixelCount})

        
        return fieldList
        
    
    def shuffle_border_score(self, min_pixel_number_per_field=20, min_peak_rate=4, min_peak_rate_proportion= 0.30, iterations=500, environment_shape = None, cm_per_bin=2, smoothing_sigma_cm=2, smoothing=True ,percentile=95):
        """
        Get a distribution of border scores that would be expected by chance for this neuron

        This uses the border_score_circular_environment() method to get the border scores

        Argument:
        min_pixel_number_per_field: minimal number of pixels for adjacent pixels above the rate threshold to be considered a field
        min_peak_rate: minimal peak rate for fields
        min_peak_rate_proportion: when adding pixels to a field, the rate needs to be higher than peak_rate*min_peak_rate_proportion
        iterations: how many shufflings to perform
        environment_shape: shape of the recording environment, should be rectangular or circular
        cm_per_bin: cm per bin in the firing rate map
        smoothing_sigma_cm: smoothing in the firing rate map
        smoothing: smoothing in the firing rate map
        percentile: percentile of the distribution of shuffled border scores that is used to get the significance threshold

        Return
        tuple: 3 1D numpy array with the border scores, CM and DM obtained by chance for this neuron
        
        
        """
        
        if environment_shape not in ["circle","rectangle","square"]:
            raise ValueError('environment_shape should be "circle" or "rectangle" or "square"')
        
        # keep a copy of the pose that we started with
        pose_at_start = self.ap.pose.copy()
        
        self.border_shuffle=np.empty((iterations))
        self.CM_shuffle=np.empty((iterations))
        self.DM_shuffle=np.empty((iterations))
      
        for i in range(iterations):
            
            self.ap.roll_pose_over_time() # shuffle the position data 
            self.firing_rate_map_2d(cm_per_bin=cm_per_bin, smoothing=smoothing, smoothing_sigma_cm=smoothing_sigma_cm) # calculate a firing rate map
            
            #res contains CM,CMHalf, DM, border_score, border_score_half, nFields
            res = self.border_score(min_pixel_number_per_field, min_peak_rate, min_peak_rate_proportion, environment_shape = environment_shape)
            # cm, dm, border_score, n_fields
            self.CM_shuffle[i] = res[0] 
            self.DM_shuffle[i] = res[1] 
            self.border_shuffle[i] = res[2] 
            self.ap.pose=pose_at_start
            
        self.border_score_threshold =  np.nanpercentile(self.border_shuffle,percentile)

        return self.border_shuffle, self.border_score_threshold, self.CM_shuffle, self.DM_shuffle
       
        
    def border_score(self, min_pixel_number_per_field=20, min_peak_rate=4, min_peak_rate_proportion= 0.30, return_field_list = False, environment_shape = None, n_wall_sections = 36, wall_section_width_radian= 2*np.pi/3):
        """
        Calculate the border score of a neuron when the animal explores a circular or rectangular environment.
        
        You should call self.firing_rate_map_2d() before calling self.border_score()
        
        Adapted from https://www.science.org/doi/suppl/10.1126/science.1166466/suppl_file/solstad.som.pdf
        
        For circular environments, instead of using 4 walls as in rectangular environment, we get 36 walls ranging 45 degrees around the circular environment and calculate CM for these 36 overlapping walls.
        This is used because most MEC border cells do not fire all around the arena but rather on one side of it.
        
        Make sure you use xy_range and set the range for the map so that the border of the environment is not at the border of the map.
        
        Border score is defined by (CM-DM)/(CM+DM), which can range from -1 to 1.
        
        CM is the highest proportion of wall pixels that are covered by the pixels of one field.
        CM is calculated for all fields and the largest CM is used to calculate the border score.
        
        DM is the mean shortest distance to the periphery for pixels that were part of a firing field, weighted by the firing rate in each pixel. 
        DM is then normalized as follows. For each pixel in the map, the shortest distance to the periphery was calculated. The largest value obtained over all map pixels was the value used for the normalization. 

        Field detection is done in c, CM and DM in python.
        

        Arguments:
        
        min_pixel_number_per_field: minimal number of pixels to be considered a field
        min_peak_rate: minimal peak firing rate within a field to perform field detection
        min_peak_rate_proporition: a pixel needs to be above peak_rate*min_peak_rate_proportion to be added to a field
        return_field_list: boolean whether to return the field list with which the border score was calculated
        environment_shape: "circle" or "rectangle" 
        n_wall_sections: number of border wall subsection that will be generated, only used for circular environments
        wall_section_width_radian: width in radian of the wall subsections, only used for circular environments
        
        Return
        
        CM, DM, border_score, number_fields
        """
 
        if environment_shape not in ["circle","rectangle","square"]:
            raise ValueError('environment_shape should be "circle" or "rectangle" or "square"')
        
        
        if not hasattr(self, 'firing_rate_map'):
            raise ValueError('Call self.firing_rate_map_2d() before calling self.border_score()')
        
        
        if self.firing_rate_map.shape != self.ap.occupancy_map.shape:
            raise ValueError('firing_rate_map {} and occupancy map {} have a different size'.format(self.firing_rate_map.shape,self.ap.occupancy_map.shape))
        
        
        # if the map has a peak firing rate of 0, it is not possible to calculate a border score
        if np.nanmax(self.firing_rate_map) == 0:
            if return_field_list:
                return (np.nan,np.nan,np.nan,0,None)
            else:
                return (np.nan,np.nan,np.nan,0)
        
        # get the firing fields of the cell, we get a list of dictionaries, each dict is a field
        fieldList = self.firing_rate_map_field_detection_fast(min_pixel_number_per_field=min_pixel_number_per_field, min_peak_rate=min_peak_rate, min_peak_rate_proportion=min_peak_rate_proportion)
        
        if len(fieldList)==0: # if there is no field, there is no border score.
            if return_field_list:
                return (np.nan,np.nan,np.nan,0,None)
            else:
                return (np.nan,np.nan,np.nan,0)
        
        # border map
        border_map = self.ap.detect_border_pixels_in_occupancy_map()
        
        if environment_shape == "circle":
            # get our series of circular wall subsections 
            border_section_maps = self.circular_border_wall_sections(border_map=border_map,n_sections = n_wall_sections , section_width_radian = wall_section_width_radian)
        elif environment_shape == "rectangle" or environment_shape == "square":
            border_section_maps = self.detect_walls_rectangular_environment(border_map=border_map)
        else:
            raise ValueError('environment_shape should be "circle" or "rectangle" or "square"')
        
        # calculate CM, CM is calculated for each field x wall section combinations, then we get the largest CM
        for field in fieldList: # get the max CM for each field
            field["CM"] = np.max([self.field_CM(field_map = field["field_map"],border_map = border_section_maps[i]) for i in range(border_section_maps.shape[0])])
             
        # get the largest CM of all fields
        CM = np.nanmax([field["CM"] for field in fieldList])
        
        # calculate DM, DM is not field by field but using all pixels that were part of a field, or all valid pixels in the firing_rate_map
        DM = self.field_DM(field_list= fieldList, border_map=border_map, rate_map=self.firing_rate_map)
        
        border_score = (CM-DM)/(CM+DM)
        
        if return_field_list:
            return CM, DM, border_score, len(fieldList), fieldList
        else:
            return CM, DM, border_score, len(fieldList)
        
    def detect_walls_rectangular_environment(self, border_map):
        """
        Function to detect the wall of a rectangular environment

        Argument:
        border_map: 2D numpy array, border pixels are set to 1, rest 0, This is the output of ap.detect_border_pixels_in_occupancy_map()

        Return:
        3D numpy array containing the 4 walls. The first dimension has a size of 4 for the 4 walls, the second and third dimensions have the same size as the border_map
        """

        midIndex = int(border_map.shape[0]/2)
        topWallIndex= np.argmax(border_map.sum(axis=1)[:midIndex])
        bottomWallIndex = np.argmax(border_map.sum(axis=1)[midIndex:])+midIndex

        midIndex = int(border_map.shape[1]/2)
        leftWallIndex= np.argmax(border_map.sum(axis=0)[:midIndex])
        rightWallIndex = np.argmax(border_map.sum(axis=0)[midIndex:])+midIndex

        xs, ys = np.meshgrid(np.arange(0,border_map.shape[1]),np.arange(0,border_map.shape[0])) # get 2 2D arrays with the x or y coordinate of the pixels

        indexMaxTop = np.expand_dims(border_map[:midIndex].argmax(axis=0),0)
        topWallMap = np.logical_and(np.logical_and(xs>leftWallIndex,xs<rightWallIndex),ys==indexMaxTop)

        indexMaxBottom = np.expand_dims(border_map.shape[0]-1-np.flip(border_map,axis=0).argmax(axis=0),0)
        bottomWallMap = np.logical_and(np.logical_and(xs>leftWallIndex,xs<rightWallIndex),ys==indexMaxBottom)

        indexMaxLeft = np.expand_dims(border_map.argmax(axis=1),1)
        leftWallMap = np.logical_and(np.logical_and(ys>topWallIndex,ys<bottomWallIndex),xs==indexMaxLeft)

        indexMaxRight = np.expand_dims(border_map.shape[1]-1-np.flip(border_map,axis=1).argmax(axis=1),1)
        rightWallMap = np.logical_and(np.logical_and(ys>topWallIndex,ys<bottomWallIndex),xs==indexMaxRight)

        return np.stack([topWallMap,bottomWallMap,leftWallMap,rightWallMap])
    
    
    
    def circular_border_wall_sections(self, border_map, n_sections = 36, section_width_radian= 2*np.pi/3):
        """
        The function returns maps in which a subsection of the wall of a circular environment.

        For example, it can create 36 wall subsection of a 45 degree width.


        Function returns a set of border_maps in which only a subset of the border pixels are set to 1. 
        Arguments:
        border_map: 2D np.array in which border pixels of a circular environment are set to 1 and rest to 0
        n_sections: Number of subsections of the environment border you want to create
        section_width_radian: width of the wall subsetions.

        Return:
        3D np.array with the first two dimensions the shape of the border_map and the third dimension the size of n_sections.
        Each border subsection is a 2D map with border pixels are set to 1 and rest to 0.
        """

        # get the coordinates of the border_pixels
        x,y = np.where(border_map)
        center = np.array([x.mean(),y.mean()])

        # we need to get the angle of each pixels in the map relative to our center.
        # to do this we need a 2d mesh
        x = np.arange(0,border_map.shape[0],1) -center[0]
        y = np.arange(0,border_map.shape[1],1) -center[1]
        xs, ys = np.meshgrid(x,y) 

        # we need to create unit vectors because we will calculate angles between vectors
        myStack = np.dstack([xs,ys]) 
        normalization = np.linalg.norm(myStack,axis=2)
        normalization[normalization==0.0] = np.nan # if there is a vector of length 0, set to invalid
        myStack = myStack/np.expand_dims(normalization,axis=2)

        # to save our maps with subsection of border wall
        res = np.empty((n_sections,border_map.shape[0],border_map.shape[1]))

        # the series of angle for which we want wall subsection
        target_angles = np.linspace(0,np.pi*2,n_sections+1)[:-1]

        # max deviation on each side of target_angles
        max_deviation = section_width_radian/2

        # loop for our target angles and get the wall subsection
        for i,target_angle in enumerate(target_angles):

            ## I know it should be cos,sin but I somehow had to .T the map deviation to get same dimension as map, the sin,cos gives me wall starting to the east
            target_direction_vector = np.array([np.sin(target_angle),np.cos(target_angle)])
            
            # get the angle between target_direction and vector in xs,ys
            exp_target_direction_vector = np.expand_dims(target_direction_vector, axis=[0,1]) # for broadcasting
            deviation_map = np.arccos(np.sum(myStack * exp_target_direction_vector,axis=2)).T
            
            if deviation_map.shape != res[i].shape:
                raise ValueError('deviation_map {} and res[i] {} have a different size'.format(deviation_map.shape,res[i].shape))

            res[i] = border_map.copy()
            res[i][deviation_map>max_deviation] = 0

        return res
        
        
    def field_CM(self,field_map,border_map):
        """
        Function to calculate the CM of a firing field. CM is used when calculating a border score. 
        
        CM is the proportion of border pixels covered by the pixels of one field.

        Arguments:
        field_map: 2D array with the pixels of the firing field set to 1 and the rest at 0. 
        border: 2D array with the pixels of the border of the environment set to 1 and the rest at 0.
        """


        if not field_map.shape==border_map.shape:
            raise ValueError('The shape of field_map and border_map is not the same')  

        nBorderPixels = np.sum(border_map)
        myStack = np.dstack([field_map,border_map]) # stack the 2 maps
        myStackSum = np.sum(myStack,axis=2) # sum the 2 maps
        nSharedPixels = np.sum(myStackSum==2) # the pixels with a sum of 2 are part of the border and field
        #print(nBorderPixels,nSharedPixels,np.nansum(field_map))
        CM = nSharedPixels/nBorderPixels
        return CM     

    

    def field_DM(self, field_list, border_map, rate_map):
        """
        Calculate DM used in the border score.

        DM is the mean shortest distance to the periphery for pixels that were part of a firing field, weighted by the firing rate in each pixel. 
        DM is then normalized as follows. For each pixel in the map, the shortest distance to the periphery was calculated. The largest value obtained over all map pixels was the value used for the normalization. 

        Arguments:
        field_list: list of dictionaries, each dictionary represent a firing field, as returned sp.firing_rate_map_field_detection_fast()
        border_map: 2D numpy array, border_map as returned by ap.detect_border_pixels_in_occupancy_map()
        rate_map: 2D numpy array, firing rate map

        Return:
        DM
        """
        if not isinstance(field_list, list):
             raise TypeError('field_list should be a list')  
            
            
        if len(field_list) == 0:
            return np.nan
        
        
        def minimalDistanceBetweenPointAndPointArray(coord,coords):
            """
            Find the minimal distance between a point and a series of points in 2 dimensions 

            This is used by field_DM

            Arguments
            coord: 1D array with x and y coordinate of a point
            coords: x by 2 array with x and y coordinates of many points
            """
            return np.min(np.sqrt((coords[:,0]-coord[0])**2 + (coords[:,1]-coord[1])**2))

        # create a map with all field pixels of all fields. Since the fields do not overlap in space, we can simply sum the stack maps to get a single map. Values of 1.0 are field pixels.
        all_fields_map = np.sum(np.dstack([field["field_map"] for field in field_list]),axis=2)

        if not all_fields_map.shape == border_map.shape:
             raise ValueError('shape of all_field_map is not the same as that of the border_map')

        # get the x,y coordinate of our pixels
        fCoord = np.squeeze(np.dstack(np.where(all_fields_map==1.0))) # field coordinates
        bCoord = np.squeeze(np.dstack(np.where(border_map==1.0))) # border coordinates
        rCoord = np.squeeze(np.dstack(np.where(~np.isnan(rate_map)))) # all valid pixels coordinates in the rate map

        # get the firing rate for field pixels
        rate_lin = rate_map[all_fields_map==1.0]  # rate of the field pixels, I am assuming that np.where give the same pixel order as this line...

        # we now have all we need for calculation of DM
        # we call a function to get the minimal distance between a point to a series of points within a np.apply_along_axis. Equivalent of two nested for loops
        # we multiply the minimal distance to border by the rate of each field bin
        # then we do a sum and divide by the sum of the firing rate of field pixels, these two last steps are the firing rate weighted mean.
        DM_not_normalized = np.sum(np.apply_along_axis(minimalDistanceBetweenPointAndPointArray,1,fCoord, bCoord)*rate_lin)/np.sum(rate_lin)
        # we get our normalization factor, which is the largest minimal distance to the border of any valid pixel of the rate map 
        distance_normalization = np.max(np.apply_along_axis(minimalDistanceBetweenPointAndPointArray,1,rCoord, bCoord))
        DM = DM_not_normalized/distance_normalization

        return DM

   
   
    

    def speed_score(self, bin_size_sec=0.02, sigma_ifr=12.5, sigma_speed=5):
        """
        Calculate the speed score of a neuron. 
        Score is calculated like in the first speed cell paper: Kropff et al. 2015 https://rdcu.be/cRaD3

        Arguments:
        For speed calculation: sigma_speed -> sigma for ap.speed_from_pose
        For ifr calculation: sigma_ifr, bin_size_sec -> sigma, bin_per_sec for ifr

        Return
        speed score
    
        """
        

        self.ap.speed_from_pose(sigma=sigma_speed)

        self.st.instantaneous_firing_rate(bin_size_sec = bin_size_sec, sigma = sigma_ifr, outside_interval_solution="nan") #kernel=250ms, binning=20ms
        ifr=self.st.ifr[0]
        if len(self.st.ifr[0])>len(self.ap.speed):
            ifr=self.st.ifr[0][0:len(self.ap.speed)]

        #remove nan
        speed=self.ap.speed[~np.isnan(self.ap.speed)&~np.isnan(ifr)]
        ifr=ifr[~np.isnan(self.ap.speed)&~np.isnan(ifr)]


        speed_score = pearsonr(speed,ifr)[0]
        
        return speed_score

        
    def shuffle_speed_score(self, iterations=500, bin_size_sec=0.02, sigma_ifr=12.5, sigma_speed=5, percentile=95):
        """
        Get a distribution of speed score that would be expected by chance for this neuron

        Argument:
        iterations: How many shufflings to perform
        see arguments of speed score
        percentile: percentile of the distribution of shuffled speed scores that is used to get the significance threshold

        Return
        tuple: 
        0: 1D numpy array with the speed scores obtained by chance for this neuron
        1: significance threshold for speed score
        
        Example
        
        # get a neuron and set intervals
        n = cg.neuron_list[7]
        n.set_spatial_properties(ap)
        n.spike_train.set_intervals(aSes.intervalDict[cond])
        n.spatial_properties.ap.set_intervals(aSes.intervalDict[cond])

        # get the observed value for border score
 
        SpS = n.spatial_properties.speed_score()

        # get the shuffled values for speed score
        shuSpS,threshold = n.spatial_properties.shuffle_speed_score(iterations=100, percentile=95, bin_size_sec=0.02, sigma_ifr=12.5, sigma_speed=5)

        # plot the results for this neuron
        res = plt.hist(shuSpS,label="shuffled")
        ymax=np.max(res[0])
        plt.plot([threshold,threshold],[0,ymax],c="black",label="Threshold")
        plt.plot([SpS,SpS],[0,ymax],c="red",label="Observed")
        plt.xlabel("Speed score")
        plt.ylabel("Count")
        plt.legend()
        plt.show()
        """
        
        # keep a copy of the pose that we started with
        pose_at_start = self.ap.pose.copy()
        
        self.speed_shuffle=np.empty(iterations)
        for i in range(iterations):
            self.ap.roll_pose_over_time() # shuffle the position data 
            # no need to recalculate the firing rate map as it will be calculated in course of border score calculation
            self.speed_shuffle[i] = self.speed_score(bin_size_sec=bin_size_sec, sigma_ifr=sigma_ifr, sigma_speed=sigma_speed) # calculate the border score from the new map
            self.ap.pose=pose_at_start

        # calculate the threshold
        if not np.isnan(self.speed_shuffle).all():
            shuffled = self.speed_shuffle[~np.isnan(self.speed_shuffle)]
        else:
            shuffled = np.nan

            
        self.speed_score_threshold =  np.percentile(shuffled,percentile)
        
        return self.speed_shuffle, self.speed_score_threshold
    

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
        
        
