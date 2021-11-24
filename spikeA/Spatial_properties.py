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
            spike_count = ndimage.gaussian_filter1d(spike_count, sigma=self.hd_histo_smoothing_sigma_deg/self.hd_histo_deg_per_bin)
    
        ## get the firing rate in Hz (spike count/ time in sec)
        self.firing_rate_head_direction_histo_edges = self.ap.hd_occupancy_bins
        self.firing_rate_head_direction_histo = spike_count/self.ap.hd_occupancy_histogram
    
    
    def head_direction_score(self):
        """
        Method to calculate the mean direction and the mean vector length from the hd-rate histogram
        
        returns a tuple: mean_direction_rad, mean_direction_deg, mean_vector_length
        """
        if not hasattr(self, 'firing_rate_head_direction_histo'):
            raise TypeError("You need to call spatial_properties.firing_rate_head_direction_histogram() before calling this function")
            
        # sum up all spikes
        sum_histo = np.sum(self.firing_rate_head_direction_histo)
        
        # if all rates are at 0, we can't calculate these scores
        if sum_histo == 0.0:
             self.hd_mean_vector_length= np.nan
        #   return np.nan
        
        # get midth of bins
        angles=self.mid_point_from_edges(self.firing_rate_head_direction_histo_edges)
        
        
        # get x and y component of each angle and multiply by firing rate
        x = np.cos(angles)*self.firing_rate_head_direction_histo
        y = np.sin(angles)*self.firing_rate_head_direction_histo
                
        
        # the angle is the arctan of x divided by y
        mean_direction = np.arctan2(np.sum(y),np.sum(x))
        
        self.hd_mean_direction_deg = mean_direction*360/(2*np.pi)
        self.hd_mean_direction_rad = mean_direction
        
        #get mean vector length
        R = np.sqrt(np.sum(x)**2+np.sum(y)**2)
        self.hd_mean_vector_length = R/sum_histo

        return (self.hd_mean_direction_rad,self.hd_mean_direction_deg, self.hd_mean_vector_length)
    
    
    def shuffle_head_direction_score(self, iterations=500, deg_per_bin=2, smoothing_sigma_deg=10, smoothing=True, percentile=95):
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
        percentile: percentile of the distribution of shuffled info scores that is used to get the significance threshold

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
            raise TypeError("The firing rate maps have different dimensions. You have to specify the xy range.")
            
        # calculate crosscorrelation
        indices = np.logical_and(~np.isnan(map1), ~np.isnan(map2))
        r,p = pearsonr(map1[indices],map2[indices])
    
        return r
