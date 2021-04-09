# Unit testing

To ensure that our code does what it is supposed to do, we need to test it. 
It is common to test a function when we first write it and then lose the code we wrote for testing. 
A better approach is to write a series of tests when developing a functionality and then keep the test to ensure no one break the code later on.

We will use `unittest` to test the functionality of `spikeA`.
The test code can be saved in the `tests` folder. 

More information about `unittest` is found [here](https://docs.python.org/3/library/unittest.html).

## Run a test

```
python -m unittest ~/repo/spikeA/tests/test_Spike_train.py
```
