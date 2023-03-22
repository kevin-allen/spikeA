#include <math.h>


void spike_triggered_occupancy_map_2d(double *spike_time, // spike times
                                      double *spike_x, // x pos of animal at spike time
                                      double *spike_y, // y pos of animal at spike time
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
                                      double valid_radius){
    /*
    Function to calculate spike-triggered occupancy maps.
    Assumes the spikes and pose data are chronologically organized
    */
    
    // occupancy map will have an odd size
    // for example 3x3, index of the center bin is 1,1
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
    
    int within_count=0; // whether we have seen a pose within the time window
    int first_within_index = 0; // index of the first pose that was within the time windown
    
     double distance = 0 ; // distance between spikes and data point of the path
    
    // loop for each spike
    for (int i = 0; i < spike_length; i++){
        // loop the pose array
        
        within_count=0;
        for (int j = first_within_index; j < pose_length;j++){
            // check if this position is within the time window after the spike
            if((time[j] >= (spike_time[i]-window_sec)) && (time[j] <= (spike_time[i]+window_sec))){
                diff_x = (int)(x[j]-spike_x[i])/cm_per_bin;
                diff_y = (int)(y[j]-spike_y[i])/cm_per_bin;
                index_x = mid_x + diff_x  ;
                index_y = mid_y + diff_y ;
                
                distance = sqrt( pow(x[j]-spike_x[i],2)+pow(y[j]-spike_y[i],2));
                
                if(distance<valid_radius){
                    if(index_x > 0 && index_x < x_bins_occ_map && index_y > 0 && index_y < y_bins_occ_map){
                        occ[index_y+ index_x*y_bins_occ_map]+= delta_time; // I reversed the indices.....????
                    }
                }
                if(within_count==0){
                    first_within_index=j;
                    within_count++;
                }
    
            }
            // assumes that the pose data are chronologically organized
            if (time[j] > (spike_time[i]+window_sec)){
                j = pose_length; // will end the loop through the pose data for this spike
            }
        
        } 
    }
}