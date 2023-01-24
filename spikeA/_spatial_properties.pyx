""" C extension to do spatial properties analysis at high speed. """

# cimport the Cython declarations for numpy
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()


# cdefine the signature of our c function
cdef extern from "spatial_properties.h":
    void map_autocorrelation(double *one_place, double *one_auto, 
                             int x_bins_place_map, int y_bins_place_map, 
                             int x_bins_auto_map, int y_bins_auto_map, 
                             int min_for_correlation)
    void map_crosscorrelation(double *one_place, double *two_place, double *one_auto, 
                             int x_bins_place_map, int y_bins_place_map, 
                             int x_bins_auto_map, int y_bins_auto_map, 
                             int min_for_correlation)
    
    double correlation (double* x, double* y, int size, double invalid)
    void detect_border_pixels_in_occupancy_map(double* occ_map, int* border_map, int num_bins_x, int num_bins_y)
    int detect_one_field(double* rate_map, int* field_map, int num_bins_x, int num_bins_y, double min_peak_rate, double min_peak_rate_proportion)
    
    

# create the wrapper code, with numpy type annotations
def map_autocorrelation_func(np.ndarray[double, ndim=2, mode="c"] one_place not None,
				             np.ndarray[double, ndim=2, mode="c"] one_auto not None):
    map_autocorrelation(<double*> np.PyArray_DATA(one_place),
				<double*> np.PyArray_DATA(one_auto),
				one_place.shape[0], one_place.shape[1],
				one_auto.shape[0], one_auto.shape[1], 5)

# create the wrapper code, with numpy type annotations
def map_crosscorrelation_func(np.ndarray[double, ndim=2, mode="c"] one_place not None,
                              np.ndarray[double, ndim=2, mode="c"] two_place not None,
				             np.ndarray[double, ndim=2, mode="c"] one_auto not None):
    map_crosscorrelation(<double*> np.PyArray_DATA(one_place),
                         <double*> np.PyArray_DATA(two_place),
				<double*> np.PyArray_DATA(one_auto),
				one_place.shape[0], one_place.shape[1],
				one_auto.shape[0], one_auto.shape[1], 5)

    
    
    
def detect_border_pixels_in_occupancy_map_func(np.ndarray[double, ndim=2, mode="c"] occ_map not None,
                                               np.ndarray[int, ndim=2, mode="c"] border_map not None):
    
    detect_border_pixels_in_occupancy_map(<double*> np.PyArray_DATA(occ_map),
                                          <int*> np.PyArray_DATA(border_map),
                                          occ_map.shape[1], occ_map.shape[0])

def detect_one_field_func(np.ndarray[double, ndim=2, mode="c"] rate_map not None,
                         np.ndarray[int, ndim=2, mode="c"] field_map not None,
                         min_peak_rate, 
                        min_peak_rate_proportion):
    
    # this is not recursive, just detect the field with the largest peak
    results = detect_one_field(<double*> np.PyArray_DATA(rate_map),
                               <int*> np.PyArray_DATA(field_map),
                               rate_map.shape[1], rate_map.shape[0],
                               min_peak_rate,
                               min_peak_rate_proportion)
    return results