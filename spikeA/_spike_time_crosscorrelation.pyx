""" C function that calculate a spike-time crosscorrelation between 2 sorted spike time arrays. It takes C double arrays as input using
    the Numpy declarations from Cython """

# cimport the Cython declarations for numpy
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()

# cdefine the signature of our c function
cdef extern from "spike_time_crosscorrelation.h":
    void spike_time_crosscorrelation(double * st1, double * st2, double* out, int size1, int size2, int outSize, double min, double max, double step)

# create the wrapper code, with numpy type annotations
def spike_time_crosscorrelation_func(np.ndarray[double, ndim=1, mode="c"] st1 not None,
    				     np.ndarray[double, ndim=1, mode="c"] st2 not None,
				     np.ndarray[double, ndim=1, mode="c"] out not None,
				     min,
				     max,
				     step):
    spike_time_crosscorrelation(<double*> np.PyArray_DATA(st1),
                                <double*> np.PyArray_DATA(st2),
				<double*> np.PyArray_DATA(out),
				st1.shape[0],
				st2.shape[0],
				out.shape[0],
				min,
				max,
				step)