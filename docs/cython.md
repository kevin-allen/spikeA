# c code with cython

We should try to avoid writing c code as much as possible as it makes it much harder to read for people not use to it. 
It is also much harder to debug as the code needs to be compiled.

But sometime c code is needed for speed. Below is an example of how I added a function to calculate spike-time crosscorrelation to the spikeA package. 
I had tried to speed it up with numba but I needed more control in the for loop.

I read an introduction to [interfacing with c](https://scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html) online.



