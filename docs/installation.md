# Installation

## Get the code

If you don't have the code, clone the repository

```
cd ~/repo
git clone https://github.com/kevin-allen/spikeA.git
```

Next time, you just need to use git pull from within the repository directory.

If you are not familiar with git, I would suggest reading a tutorial online.

## Use your python environment

If you don't already have a python environment for your data analysis, create one.

```
conda deactivate
python3 -m venv /home/kevin/python_virtual_environments/skikeAenv
source /home/kevin/python_virtual_environments/skikeAenv/bin/activate
```

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
from spikeA.Spike_train import Spike_train
Spike_train("hey")
```
If there is no error, `spikeA` is now installed in your environment.
