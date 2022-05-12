import os
import pydicom as dicom

from General.Configuration import aapmMayoConfiguration, experimentConfiguration

class DataHandler:
    '''
        The class to handle various data loading and HU reconstruction
    '''
    def __init__(self, experimentConfiguration, loadingConfiguration):
        self.experimentConfiguration=experimentConfiguration
        self.loadingConfiguration=loadingConfiguration

    
    def generateAapmMayo(self):
        '''
            Generate text files where each one of them contains
        all the absolute paths for loading dicom files
        '''
        patientAll=os.listdir(self.loadingConfiguration.path)

        LRfolder=tuple(sorted((self.loadingConfiguration.path+"/"+patient+"/"+patientFolder for patient in patientAll
                                                                        for patientFolder in os.listdir(self.loadingConfiguration.path+"/"+patient)
                                                                        if "Low" in patientFolder)))

        HRfolder=tuple(sorted((self.loadingConfiguration.path+"/"+patient+"/"+patientFolder for patient in patientAll
                                                                        for patientFolder in os.listdir(self.loadingConfiguration.path+"/"+patient)
                                                                        if "Full" in patientFolder)))

        for i in range(len(LRfolder)):                                                            
            name=f"LR {i}.txt"
            content=tuple(sorted((LRfolder[i]+"/"+image for image in os.listdir(LRfolder[i]))))
            with open(self.experimentConfiguration.projectPath+"/"+"AapmMayo"+"/"+name, "w") as file:
                for line in content:
                    file.write(line+"\n")

        for i in range(len(HRfolder)):                                                            
            name=f"HR {i}.txt"
            content=tuple(sorted((HRfolder[i]+"/"+image for image in os.listdir(HRfolder[i]))))
            with open(self.experimentConfiguration.projectPath+"/"+"AapmMayo"+"/"+name, "w") as file:
                for line in content:
                    file.write(line+"\n")

    def loadAapmMayo2D(self):
        '''
            Load the dicom files from Aapm Mayo dataset with given resolution
        for training 2D model

            In this mode, files from different patients are stored together and
        returned as tuple
        '''
        result=[]
        patientAll=tuple(sorted((self.experimentConfiguration.projectPath+"/"+"AapmMayo"+"/"+patient for patient in os.listdir(self.experimentConfiguration.projectPath+"/"+"AapmMayo")
                        if self.loadingConfiguration.resolution in patient)))
        
        for patient in patientAll:
            with open(patient, "r") as file:
                content=file.readlines()
            for line in content:
                result.append(line.strip("\n"))
        
        return tuple(sorted(result))
    
    def loadAapmMayo3D(self):
        '''
            Load the dicom files from Aapm Mayo dataset with given resolution
        for training 3D model

            In this mode, files within the same folder are stored together and
        returned as dict
        '''
        result={}
        patientAll=tuple(sorted((self.experimentConfiguration.projectPath+"/"+"AapmMayo"+"/"+patient for patient in os.listdir(self.experimentConfiguration.projectPath+"/"+"AapmMayo")
                if self.loadingConfiguration.resolution in patient)))
        
        for patient in patientAll:
            with open(patient, "r") as file:
                content=file.readlines()
            contentStrip=tuple(sorted((line.strip("\n") for line in content)))
            result[patient]=contentStrip

        return result
        




    


    
