void spike_triggered_occupancy_map_2d(double *spike_time, // spike times
                                      double *spike_x, // x pos of animal at spike time
                                      double *spike_y, // y pos of animal at spike time
                                      int spike_length, // number of spikes
                                      double *time, // time for the position data
                                      double *x, // x position of animal
                                      double *y, // y position of animal
                                      int pose_length, // length of position data arrays
                                      double window_sec, // time to considered after each spike
                                      double *occ, // occupancy map
                                      int x_bins_occ_map,
                                      int y_bins_occ_map,
                                      double cm_per_bin){
    /*
    Function to calculate spike-triggered occupancy maps.
    */
    
    int mid_x = x_bins_occ_map/2; // mid point of the occupancy map
    int mid_y = y_bins_occ_map/2; // mid point of the occupancy map
    
    double diff_x;
    double diff_y;
    
    int index_x;
    int index_y;
    
    double delta_time = time[1]-time[0];
     
    // set occ map to 0.0
    for(int x = 0; x < x_bins_occ_map; x++){
        for(int y = 0; y < y_bins_occ_map; y++){
            occ[x+y*x_bins_occ_map] = 0.0;
        }
    }
        
    // loop for each spike
    for (int i = 0; i < spike_length; i++){
        // loop the pose array
        for (int j = 0; j < pose_length;j++){
            // check if this position is within the time window after the spike
            if(time[j] >= spike_time[i] && time[j] <= spike_time[i]+window_sec){
                diff_x = (int)(x[j]-spike_x[i])/cm_per_bin;
                diff_y = (int)(y[j]-spike_y[i])/cm_per_bin;
                index_x = mid_x + diff_x;
                index_y = mid_y + diff_y;
                if(index_x > 0 && index_x < x_bins_occ_map && index_y > 0 && index_y < y_bins_occ_map){
                    occ[index_y+ index_x*y_bins_occ_map]+= delta_time; // I reversed the indices.....????
                }
            } 
        }
        
        
    }
    
    
    
}