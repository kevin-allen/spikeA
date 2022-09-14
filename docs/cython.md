# c code with cython

In general, we should try to avoid writing c code as much as possible as it makes it much harder to read for most user and it is harder to debug.

But sometime c code is needed for speed. Below is an example of how I added a function to calculate spike-time crosscorrelation to the spikeA package. 
I had tried to speed it up with numba but I needed more control in the for loop.

I read an introduction to [interfacing with c](https://scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html) online.

Based on this introduction, I decided to use [cython](https://cython.org/) to interface with c code. The documentation is found [here](https://cython.readthedocs.io/en/latest/).

Cython has its own language that I did not learn. 

* [Interactions with c libraries and Cython](https://cython.readthedocs.io/en/latest/src/tutorial/clibraries.html)
* [Working with NumPy array and Cython](http://docs.cython.org/en/latest/src/tutorial/numpy.html)



## Installation

You might have to install cython in your python environment.

```
conda install -c anaconda cython
```
or
```
pip install Cython
```

## Using a c function in python

There are a few steps required to use a c function in python.

1. Declare the function in a `.h` file.
2. Write the function in a `.c` file.
3. Import the function in a `.pyx` file and create a wrapper function. The `.pyx` file starts with an `_`
4. Compile c code to a library (module)
5. Import the module and call the function from python

Below is a step-by-step example of how I created a c function to computer the spike-time crosscorrelation between 2 arrays of spike times.


### Declare the function in a .h file

In `spikeA/spikeA/spike_time.h`
```
void spike_time_crosscorrelation(double * st1, double * st2, double* out, int size1, int size2, int outSize, double min, double max, double step);
```
This is plain c code.

### Write the function in a .c file

In `spikeA/spikeA/spike_time.c`
```

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
```

### Import the function in a .pyx file and create a wrapper function

In `spikeA/spikeA/_spike_time.pyx`, you need to import your c function and create a wrapper for python

See the file for more information

### Compile c code to a library (module)

In `spikeA/spikeA/setup.py`, tell cpython how to compile our new module.

From the directory `spikeA/spikeA`, run 

```
python setup.py build_ext -i
```

### Import the module and call the function from python

In a jupyter notebook, you can try the following

```
import numpy as np
import matplotlib.pyplot as plt
import spikeA.spike_time # no capital (not the python class but the c extension)

?spikeA.spike_time
```

```
dir(spikeA.spike_time)
```

```
x = np.arange(0, 2 * np.pi, 0.1)
y = np.empty_like(x)
out = np.empty(10)
## this is not gonna work!
spikeA.spike_time.spike_time_crosscorrelation_func(x,y,out,-0.1,0.1,0.0005)
out
```
