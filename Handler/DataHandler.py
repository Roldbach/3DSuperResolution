import os

from General.Configuration import aapmMayoConfiguration

class DataHandler:
    '''
        The class to handle various data loading and HU reconstruction
    '''
    def __init__(self, name, loadingConfiguration):
        self.name=name
        self.loadingConfiguration=loadingConfiguration

    
    def loadAapmMayoHR3D(self):
        '''
            Load Aapm Mayo dataset depending on the mode and resolution
        '''
        patientAll=os.listdir(self.loadingConfiguration.path)
        folder=tuple(sorted((self.loadingConfiguration.path+"/"+patient+"/"+patientFolder for patient in patientAll
                                                                        for patientFolder in os.listdir(self.loadingConfiguration.path+"/"+patient)
                                                                        if "Full" in patientFolder)))                                                            
        
    


    
