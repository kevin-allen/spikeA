# Download one recording session

In order to test the spikeA package with real data, we uploaded one recording session from a mouse with tetrodes in the entorhinal cortex.

There is a function in spikeA to download it to your computer. Just run the following code.

```
from spikeA.Downloads import download_spikeA_data_one_session
download_spikeA_data_one_session()
```

If you are beind a proxy, make sure you have the environment variable https_proxy defined in your shell.
```
export https_proxy=www-int2.inet.dkfz-heidelberg.de:80
```

If the function can't download the file for some reasons, the link to the file is in the docstring of the function.

```
?download_spikeA_data_one_session
```

Alternatively click [here](https://drive.google.com/file/d/1xq3wx-k8hv7oLKQqcjoiXxn7aWhwS_6B/view?usp=sharing). If you download it with your browser, you will need to extract the compressed file.

