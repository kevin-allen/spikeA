/*  Function to calculate the spike-time crosscorrelation between two arrays containing sorted spike times. */
#include <stdio.h>


void spike_time_crosscorrelation(double * st1, double * st2, double* out, int size1, int size2, int outSize, double min, double max, double step){
    int i;
    int j;
    int prev_start = 0;
    int next_start = 0;
    double time_diff = 0;
    int index = 0;
    
    for(i=0;i<size1;i++){ // for all reference spikes
        for(j=prev_start; j < size2; j++){ // spikes in st2
            
            time_diff = st2[j] - st1[i];
            if(time_diff < min){prev_start=j;} // next ref spike, no need to consider this spike
            if(time_diff > max){j=size2;} // done for this reference spike because we are too far in st2
            if(time_diff>min && time_diff<max){ // add this spike to the out histogram
                index = (int)((time_diff-min)/step); 
                out[index]++;
            }
        
        }
    }
}

