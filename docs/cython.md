# c code with cython

We should try to avoid writing c code as much as possible as it makes it much harder to read for people not use to it. 
It is also much harder to debug as the code needs to be compiled.

But sometime c code is needed for speed. Below is an example of how I added a function to calculate spike-time crosscorrelation to the spikeA package. 
I had tried to speed it up with numba but I needed more control in the for loop.

I read an introduction to [interfacing with c](https://scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html) online.

Based on this introduction, I decided to use [cython](https://cython.org/) to interface with c code. The documentation is found [here](https://cython.readthedocs.io/en/latest/).

cython has its own language that I did not learn. 


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

1. Declare the function in a .h file.
2. Write the function in a .c file.
3. Import the function in a .pyx file and create a wrapper function
4. Compile c code to a library (module)
5. Import the module and call the function from python

Below is a step-by-step example of how I created a c function to computer the spike-time crosscorrelation between 2 arrays of spike times.


### Declare the function in a .h file

In `spikeA/spikeA/spike_time_crosscorrelation.h`
```
void spike_time_crosscorrelation(double * st1, double * st2, double* out, int size1, int size2, int outSize, double min, double max, double step);
```
This is plain c code.

### Write the function in a .c file

In `spikeA/spikeA/spike_time_crosscorrelation.c`
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

In `spikeA/spikeA/_spike_time_crosscorrelation.pyx`, you need to import your c function and create a wrapper for python

```
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
```

### Compile c code to a library (module)

In `spikeA/spikeA/setup.py`, tell cpython how to compile our new module.

```
from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("spike_time_crosscorrelation",
                 sources=["_spike_time_crosscorrelation.pyx", "spike_time_crosscorrelation.c"],
                 include_dirs=[numpy.get_include()])],
)
```

Then run 

```
python setup.py build_ext -i
```

### Import the module and call the function from python

In a jupyter notebook, you can try the following

```
import numpy as np
import matplotlib.pyplot as plt
import spikeA.spike_time_crosscorrelation

?spikeA.spike_time_crosscorrelation
```

```
dir(spikeA.spike_time_crosscorrelation)
```

```
x = np.arange(0, 2 * np.pi, 0.1)
y = np.empty_like(x)
out = np.empty(10)

spikeA.spike_time_crosscorrelation.spike_time_crosscorrelation_func(x,y,out,-0.1,0.1,0.0005)
out
```
