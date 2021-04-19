# For developers

We will try to develop spikeA as a collaborative project within the lab. We will have to establish a few guidlines to make sure the code is consistant.


## What you should know

### Git

The code for spikeA is stored in a GitHub repository. 
This means that you will need to use `git` to get the code and to update the repository with your changes. 
Try to learn a bit about how `git` is used. In the end, you will only need to use a few commands to do 95% of what you need (e.g., clone, commit, status, pull, push, etc.).

### Python

Most of the code will be in python. This means that some knowledge of python will be required. 
We will use the main python packages used for scientific computing: `numpy`, `scipy`, `pandas`, `matplotlib` and `scikit-learn`. 
These package are well-established and you can find usefull tutoirals and help online.
If you are not familiar with python, you might want to do a small online course before getting going.

Alternatively, you can read [`Python for Data Analysis`](https://www.oreilly.com/library/view/python-for-data/9781491957653/) by Wes McKinney.

### Python environment

I would recommand to create a python environment to do your data analysis. More on this in how to install and use spikeA.

## Coding guidelines

1. Organize your code so that it is as readable as possible. See [Python style guidelines](https://www.python.org/dev/peps/pep-0008/).
2. Class names are Captalized.
3. For attributes and methods: lowercase with words separated by underscores.
4. Document all Class, methods and function with a [docstring](https://www.python.org/dev/peps/pep-0257/).

## Adding new functionalities

When adding new functionalities to spikeA, you should provide the following.

1. Write a brief description of what your code is doing (for docstring).
2. Provide some examples of how to use this code (for docstring).
3. If possible, provide an example in which you visualize the input and output of your code (for docstring).
4. Try to write some tests to make sure that your code is doing what you think it is doing (for tests directory).

