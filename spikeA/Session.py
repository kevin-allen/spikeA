import os.path
import os
import pandas as pd
import numpy as np
from datetime import datetime
class Session:
    """
    Class containing information about a recording session
    
    Should have a way to store information about electrode types, electrode location, should work for probes and tetrodes
    We will need a calss that deals with datfiles (similar to relectro)
    We will need a Cell_group class that deals with Neuron objects.
    The Neuron class would contain a spike train.
    
    Attributes:
        name: Name of the session. Usually used as the beginning of the file names. Format is assumed to be subject-date-time
        path: Directory path of the data for this session
        subject: Name of the subect. This assumes that the session name is in the format subject-date-time
        sessionDateTime: Datetime of the session. This assumes that the session name is in the format subject-date-time
        fileBase: path + name
        requiredFileExts: List containing the extensions of the file we should have in the directory
        dataFileCheck: Boolean indicating whether to test for the presence of data file in the session directory
        fileNames: Dictionary to get important file names
        
    Methods:
        checkSessionDirectory()
    """
    def __init__(self,name, path, dataFileCheck=True):
        self.name = name
        self.path = path
        self.subject = self.name.split("-")[0]
        self.sessionDateTime = datetime.strptime(self.name.split("-")[1]+self.name.split("-")[2], '%d%m%Y%H%M')
        self.fileBase = path+"/"+name
        self.requiredFileExts = ["par","desen","desel"]
        
        # check that we have valid data
        self.dirOk=False
        if dataFileCheck:
            if self.checkSessionDirectory():
                self.dirOk=True
            else:
                print("problem with the directory " + self.path)

        #####################################################
        # create a dictonary to quickly get the file names ##
        # easier to get the file names                     ##
        #####################################################
        self.fileNames = {"par": self.fileBase+".par",
                         "desen": self.fileBase+".desen",
                         "desel": self.fileBase+".desel",
                         "resofs": self.fileBase+".resofs",
                         "pxPerCm": self.fileBase+".pxPerCm",
                         "samplingRAte": self.fileBase+".samplingRate"}
        return

    def checkSessionDirectory(self):
        # check that the directory is there
        if os.path.isdir(self.path) == False :
            raise IOError(self.path + " does not exist") # raise an exception
            
        # check that the files needed are there
        for ext in self.requiredFileExts:
            fileName = self.fileBase + "." + ext
            if os.path.isfile(fileName)== False:
                print(fileName + " does not exist")
                raise IOError(fileName + " does not exist") # raise an exception
                
        # check if the dat files are there
        
        # check if the positrack files are there
        
        return True
    