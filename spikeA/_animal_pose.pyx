# cimport the Cython declarations for numpy
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()


# cdefine the signature of our c function
cdef extern from "animal_pose.h":
    void spike_triggered_occupancy_map_2d(double *spike_time,
                                          double *spike_x,
                                          double *spike_y,
                                      int spike_length,
                                      double *time, 
                                      double *x,
                                      double *y, 
                                      int pose_length, 
                                      double window_sec, 
                                      double *occ, 
                                      int x_bins_occ_map,
                                      int y_bins_occ_map,
                                      double cm_per_bin,
                                      double valid_radius)
    
    
# create the wrapper code, with numpy type annotations
def spike_triggered_occupancy_map_2d_func(np.ndarray[double, ndim=1, mode="c"] spike_time not None,
                                          np.ndarray[double, ndim=1, mode="c"] spike_x not None,
                                          np.ndarray[double, ndim=1, mode="c"] spike_y not None,
                                          np.ndarray[double, ndim=1, mode="c"] time not None,
                                          np.ndarray[double, ndim=1, mode="c"] x not None,
                                          np.ndarray[double, ndim=1, mode="c"] y not None,
                                          window_sec,
                                          np.ndarray[double, ndim=2, mode="c"] occ not None,
                                          cm_per_bin,
                                          valid_radius):
        
    spike_triggered_occupancy_map_2d(<double*> np.PyArray_DATA(spike_time),
                                     <double*> np.PyArray_DATA(spike_x),
                                     <double*> np.PyArray_DATA(spike_y),
                                      spike_time.shape[0],
                                      <double*> np.PyArray_DATA(time),
                                      <double*> np.PyArray_DATA(x),
                                      <double*> np.PyArray_DATA(y),
                                      time.shape[0],
                                      window_sec,
                                      <double*> np.PyArray_DATA(occ),
                                      occ.shape[0],occ.shape[1],cm_per_bin,valid_radius)
                                     