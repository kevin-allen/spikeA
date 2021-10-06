""" C extension to do spike-time analysis at high speed. """

# cimport the Cython declarations for numpy
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()

# cdefine the signature of our c function
cdef extern from "spatial_properties.h":
    void map_autocorrelation(double *one_place, double *one_auto, int x_bins_place_map, int y_bins_place_map, int x_bins_auto_map, int y_bins_auto_map, int min_for_correlation)
    

# create the wrapper code, with numpy type annotations
def spatial_properties_func(np.ndarray[double, ndim=1, mode="c"] one_place not None,
				     np.ndarray[double, ndim=1, mode="c"] one_auto not None):
    spatial_properties(<double*> np.PyArray_DATA(one_place),
				<double*> np.PyArray_DATA(one_auto),
				one_place.shape[0],
				one_auto.shape[0])
