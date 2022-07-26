/*  Function to calculate the spike-time crosscorrelation between two arrays containing sorted spike times. */
#include <stdio.h>


void spike_time_autocorrelation(double * st, double* out, int inSize, int outSize, double min, double max, double step){
    int i;
    int j;
    int prev_start = 0;
    double time_diff = 0;
    int index = 0;
    
    for(i=0;i<inSize;i++){ // for all reference spikes
        for(j=prev_start; j < inSize; j++){ // for spikes around the reference spike
            
            if (i != j){
                time_diff = st[j] - st[i];
                if(time_diff < min){prev_start=j;} // next ref spike, no need to consider this spike
                if(time_diff > max){j=inSize;} // done for this reference spike because we are too far in st2
                if(time_diff>min && time_diff<max){ // add this spike to the out histogram
                    index = (int)((time_diff-min)/step); 
                    out[index]++;
                }
            }
        }
    }
}

void spike_time_crosscorrelation(double * st1, double * st2, double* out, int size1, int size2, int outSize, double min, double max, double step){
    int i;
    int j;
    int prev_start = 0;
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

void spike_phase(double* st, double* cycle_start, double* cycle_end, double* out, int stSize, int cycleSize){
    /*
    calculate the phase of spikes
    
    Returned values are -1.0 for invalid
    Phase goes from 0 to 2*pi
    */
    // set to an invalid value by default
    for (int i = 0; i < stSize; i++){
        out[i] = -1.0;
    }
    
    // find phase of spikes within a cycle
    for (int i = 0; i < stSize; i++){
        for(int j =0; j < cycleSize; j++){
            if(st[i]>cycle_start[j]&& st[i]<cycle_end[j])
            {
                // = proportion of the cycle * 2*pi 
                out[i] = (st[i]-cycle_start[j])/ (cycle_end[j]-cycle_start[j]) * 3.141592653589793238 * 2;
                j=cycleSize; // move to next spike
                    
            }
        }
    }
}
    