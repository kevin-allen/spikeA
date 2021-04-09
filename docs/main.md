# spikeA

This is the documentation for the python package spikeA. This package can be used to perform data analysis of spike trains in python.

## Clone the repository

You can download the code to your computer using the 

```
cd ~/repo
git clone https://github.com/kevin-allen/spikeA.git
```
Next time, you just need to use git pull from within the repository directory.

## Installing the repository 

First launch your python environment. Then install the `spikeA` package in your environment using pip.

```
pip install -e ~/repo/spikeA
```

You should then be able to import the different modules of the package from ipython

To get a ipython terminal
```
ipython
```

Then to load the spikeA.SpikeTrain module, run this within ipython
```
from spikeA.SpikeTrain import SpikeTrain
SpikeTrain("hey")
```


