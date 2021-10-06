""" C extension to do spatial properties analysis at high speed. """

# cimport the Cython declarations for numpy
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()

# cdefine the signature of our c function
cdef extern from "spatial_properties.h":
    void map_autocorrelation(double *one_place, double *one_auto, int x_bins_place_map, int y_bins_place_map, int x_bins_auto_map, int y_bins_auto_map, int min_for_correlation)
    

# create the wrapper code, with numpy type annotations
def map_autocorrelation_func(np.ndarray[double, ndim=2, mode="c"] one_place not None,
				     np.ndarray[double, ndim=2, mode="c"] one_auto not None):
    map_autocorrelation(<double*> np.PyArray_DATA(one_place),
				<double*> np.PyArray_DATA(one_auto),
				one_place.shape[0], one_place.shape[1],
				one_auto.shape[0], one_auto.shape[1], 1)

#    # cdefine the signature of our c function
#cdef extern from "spike_time.h":
#    void spike_time_autocorrelation(double * st, double* out, int inSize, int outSize, double min, double max, double step)
#    void spike_time_crosscorrelation(double * st1, double * st2, double* out, int size1, int size2, int outSize, double min, double max, double step)
#    
#
## create the wrapper code, with numpy type annotations
#def spike_time_autocorrelation_func(np.ndarray[double, ndim=1, mode="c"] st not None,
#				     np.ndarray[double, ndim=1, mode="c"] out not None,
#				     min,
#				     max,
#				     step):
#    spike_time_autocorrelation(<double*> np.PyArray_DATA(st),
#				<double*> np.PyArray_DATA(out),
#				st.shape[0],
#				out.shape[0],
#				min,
#				max,
#				step)
#				
#    
#    
#    
## create the wrapper code, with numpy type annotations
#def spike_time_crosscorrelation_func(np.ndarray[double, ndim=1, mode="c"] st1 not None,
#    				     np.ndarray[double, ndim=1, mode="c"] st2 not None,
#				     np.ndarray[double, ndim=1, mode="c"] out not None,
#				     min,
#				     max,
#				     step):
#    spike_time_crosscorrelation(<double*> np.PyArray_DATA(st1),
#                                <double*> np.PyArray_DATA(st2),
#				<double*> np.PyArray_DATA(out),
#				st1.shape[0],
#				st2.shape[0],
#				out.shape[0],
#				min,
#				max,
#				step)
#				