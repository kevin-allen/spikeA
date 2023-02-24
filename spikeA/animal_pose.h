void spike_triggered_occupancy_map_2d(double *spike_time, // spike times
                                      double *spike_x,
                                      double *spike_y,
                                      int spike_length, // number of spikes
                                      double *time, // time for the position data
                                      double *x, // x position of animal
                                      double *y, // y position of animal
                                      int pose_length, // length of position data arrays
                                      double window_sec, // time to considered before and after each spike
                                      double *occ, // occupancy map
                                      int x_bins_occ_map,
                                      int y_bins_occ_map,
                                      double cm_per_bin,
                                      double valid_radius);