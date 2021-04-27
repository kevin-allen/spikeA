import os.path
import os
import pandas as pd
import numpy as np
from datetime import datetime
class Session:
    """
    Class containing information about a recording session.
    
    This class is very generic and should work with all types of recording sessions.
    
    We will derive more specific class from it that will deal with the more specific stuff.
    
    Attributes:
        name: Name of the session. Usually used as the beginning of the file names. Format should be subject-date-time
        path: Directory path of the data for this session. Should not end with a /
        subject: Name of the subect. This assumes that the session name is in the format subject-date-time
        session_date_time: Datetime of the session. This assumes that the session name is in the format subject-date-time
        file_base: path + name
        
    Methods:
        
    """
    def __init__(self,name, path):
        self.name = name
        self.path = path
        self.subject = self.name.split("-")[0]
        self.sessionDateTime = datetime.strptime(self.name.split("-")[1]+self.name.split("-")[2], '%d%m%Y%H%M')
        self.fileBase = path+"/"+name
        return

    
class TetrodeSession(Session):
    """
    Class containing information about a recording session in which tetrodes were used
    """
    pass

class ProbeSession(Session):
    """
    Class containing information about a recording session in which Neuropixels or Neuronexus probes were used
    """
    pass