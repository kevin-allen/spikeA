# Installation

## Get the code

If you don't have the code, clone the repository

```
cd ~/repo
git clone https://github.com/kevin-allen/spikeA.git
```

Next time, you just need to use git pull from within the repository directory.

If you are not familiar with git, I would suggest reading a tutorial online.

## Install spikeA in your python environment

You should do your data analysis from a python environment. If you don't know what a virtual environment is read [this](https://docs.python.org/3/library/venv.html#venv-def).

Make sure your python environment is active. Then install the `spikeA` package in your environment using pip.

```
pip install -e ~/repo/spikeA
```

You should then be able to import the different modules of the package from ipython

## Test your installation

To get a ipython terminal
```
ipython
```

Then to load the spikeA.SpikeTrain module, run this within ipython
```
from spikeA.SpikeTrain import SpikeTrain
SpikeTrain("hey")
```
If there is no error, `spikeA` is now installed in your environment.