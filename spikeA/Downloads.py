import requests
import os.path
import tarfile
import tempfile 
#import gdown


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_spikeA_data_one_session(destination=None):
    """
    Function to download the data of a single recording session from the internet
    
    The data were processed with Klustakwik
    
    If the download fail, you can download it manually at :  https://drive.google.com/file/d/1xq3wx-k8hv7oLKQqcjoiXxn7aWhwS_6B/view?usp=sharing
    
    Arguments:
    destination: directory where to save the data. By default, the data will go to your home directory.
    """
    file_id = '1xq3wx-k8hv7oLKQqcjoiXxn7aWhwS_6B'
    with tempfile.TemporaryDirectory() as tmpdirname:
        #tmpdirname="/tmp"
        print('Created temporary directory', tmpdirname)
        tmp_destination = tmpdirname+"/"+"mn1385-16032021-0101.tar.gz"
        print("Downloading compressed file to",tmp_destination)
        
        # This uses requests to download the file
        download_file_from_google_drive(file_id, tmp_destination)
        
        # This uses gdown, better error message but is an additional requirement.
        #url = 'https://drive.google.com/uc?id=1xq3wx-k8hv7oLKQqcjoiXxn7aWhwS_6B'
        #gdown.download(url, tmp_destination, quiet=False)
        
        # get file size
        f_size = os.path.getsize(tmp_destination)
        print("Size of downloaded file: {}".format(f_size))
        expected_size = 2412591360
        if(f_size < 2412591360):
            print("Problem downloading the file")
            print("It might be that too many people tried to download it in a short time period.")
            print("You can try to download it with this link:")
            print("https://drive.google.com/file/d/1xq3wx-k8hv7oLKQqcjoiXxn7aWhwS_6B/view?usp=sharing")
            raise OSError("Problem with downloading the data file.")
        
        
        if destination is None:
            destination = os.path.expanduser("~")
        
        if not os.path.exists(destination):
            raise OSError("directory is missing:"+destination)
        
        print("Opening the archive",tmp_destination)
        tar = tarfile.open(tmp_destination, 'r:gz')
        main_dir = os.path.commonprefix(tar.getnames())
        print("Extracting data to {}/{}".format(destination,main_dir))
        tar.extractall(destination)
    print("Your data are in {}/{}.".format(destination,main_dir))