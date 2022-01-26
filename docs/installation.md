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
mkdir ~/python_virtual_environments
python3 -m venv ~/python_virtual_environments/spikeAenv
source ~/python_virtual_environments/spikeAenv/bin/activate
```
## Update pip

```
pip install --upgrade pip
```


## Install spikeA in your python environment

You should do your data analysis from a python environment. If you don't know what a virtual environment is read [this](https://docs.python.org/3/library/venv.html#venv-def).

Make sure your python environment is active. Then install the `spikeA` package in your environment using pip.

```
cd ~/repo/spikeA
pip install -r requirements.txt
pip install -e ~/repo/spikeA
```

If this stalls, check that your proxy is set properly. For instance at the DKFZ, you would use this code.
```
echo $https_proxy
export https_proxy=www-int2.inet.dkfz-heidelberg.de:80
```

You should then be able to import the different modules of the package from ipython

## Compile some Cython libraries

I could not find a way to compile the cython code during installation. So you will need to do it.

```
cd ~/repo/spikeA/spikeA/
python setup.py build_ext --inplace
```

This will create a module that you will be able to load from python.

## Install ipython

```
pip install jupyterlab
```

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

If some requirements are missing, try this

```
cd ~/repo/spikeA
pip install -r requirements.txt
```
