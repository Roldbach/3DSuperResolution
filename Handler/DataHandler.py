import numpy as np
import os
import pydicom as dicom

from General.Configuration import experimentConfiguration
from General.DataLoading import truncateSlice
from General.ImageProcessing import patchExtraction2D, patchExtraction3D, windowing

class DataHandler:
    '''
        The class to handle various data loading and HU reconstruction
    '''
    def __init__(self, loadingConfiguration, experimentConfiguration=experimentConfiguration):
        self.loadingConfiguration=loadingConfiguration
        self.experimentConfiguration=experimentConfiguration

        self.slice=self.loadSlice()
        self.image=self.loadImage()
        self.train, self.validation, self.test=self.splitTrainValidation()
        self.patchExtraction()
        
    def generateAapmMayoPath(self):
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

    def loadSlice(self):
        '''
            Load slice dataset according to the loading configuration

            During the processing of loading slices, the resolution will be
        automatically selected according to the configuration
        '''
        if "Aapm" in self.loadingConfiguration.path:
            if self.loadingConfiguration.mode=="2D":
                return self.loadAapmMayoSlice2D()
            else:
                return self.loadAapmMayoSlice3D()

    def loadAapmMayoSlice2D(self):
        '''
            Load the dicom files from Aapm Mayo dataset with given resolution
        for training 2D model

            In this mode, files from different patients are stored together
        
        return:
            result: dict, key=file path, value=corresponding dicom file
        '''
        result={}
        patientAll=tuple(sorted((self.experimentConfiguration.projectPath+"/"+"AapmMayo"+"/"+patient for patient in os.listdir(self.experimentConfiguration.projectPath+"/"+"AapmMayo")
                        if self.loadingConfiguration.resolution in patient)))
        
        for patient in patientAll:
            with open(patient, "r") as file:
                content=file.readlines()
            for line in content:
                result[line.strip("\n")]=dicom.read_file(line.strip("\n"))
        
        return result
    
    def loadAapmMayoSlice3D(self):
        '''
            Load the dicom files from Aapm Mayo dataset with given resolution
        for training 3D model

            In this mode, files within the same folder are stored together
        
        return:
            result: dict, key=patient name, value=tuple, contains all dicom files
        '''
        result={}
        patientAll=tuple(sorted((self.experimentConfiguration.projectPath+"/"+"AapmMayo"+"/"+patient for patient in os.listdir(self.experimentConfiguration.projectPath+"/"+"AapmMayo")
                if self.loadingConfiguration.resolution in patient)))
        
        for patient in patientAll:
            with open(patient, "r") as file:
                content=file.readlines()
            contentStrip=tuple(sorted((line.strip("\n") for line in content)))
            contentTruncate=truncateSlice(contentStrip, self.experimentConfiguration.patchWindowShape[0])
            result[patient]=[dicom.read_file(slice) for slice in contentTruncate]
        
        return result

    def loadImage(self):
        '''
            Load image dataset according to the loading configuration
        '''
        if self.loadingConfiguration.mode=="2D":
            return self.loadImage2D()
        else:
            return self.loadImage3D()

    def loadImage2D(self):
        '''
            Load images under 2D mode and apply
        windowing to each of them

        return:
            result: dict, key=file path, value=2-D image
        '''
        result={}
        for key, value in self.slice.items():
            result[key]=windowing(value)
        return result
    
    def loadImage3D(self):
        '''
            Load images under 3D mode and apply
        windowing to each of them

        return:
            result: dict, key=patient name, value=3-D image
        '''
        result={}
        for key, value in self.slice.items():
            image=[windowing(file) for file in value]
            result[key]=np.stack(image).astype("float32")
        return result
    
    def splitTrainValidation(self):
        '''
            Split the slice dataset into train, validation and test dataset
        according to the given proportions
        
        return:
            train: list, contains all images in the training dataset
            validation: list, contains all images in the validation dataset
            test: list, contains all images in the testing dataset
        '''
        keys=tuple(self.slice.keys())

        trainLimit=int(len(keys)*self.loadingConfiguration.trainProportion)
        validationLimit=int(len(keys)*self.loadingConfiguration.trainProportion)
        
        keyTrain=keys[len(keys)-trainLimit:len(keys)]
        keyValidation=keys[len(keys)-trainLimit-validationLimit:len(keys)-trainLimit]
        keyTest=keys[:len(keys)-trainLimit-validationLimit]

        train=[self.image[key] for key in keyTrain]
        validation=[self.image[key] for key in keyValidation]
        test=[self.image[key] for key in keyTest]

        return train, validation, test
    
    def patchExtraction(self):
        '''
            Patch the train and validation according to the experiment configuration

            If no valid patch window given, convert the dataset to np.array for training
        '''
        if len(self.experimentConfiguration.patchWindowShape)==3:
            self.train=np.array(patchExtraction3D(self.train, self.experimentConfiguration.patchWindowShape))
            self.validation=np.array(patchExtraction3D(self.validation, self.experimentConfiguration.patchWindowShape))
        elif len(self.experimentConfiguration.patchWindowShape)==2:
            self.train=np.array(patchExtraction2D(self.train, self.experimentConfiguration.patchWindowShape, self.experimentConfiguration.patchStep))
            self.validation=np.array(patchExtraction2D(self.train, self.experimentConfiguration.patchWindowShape, self.experimentConfiguration.patchStep))
        else:
            self.train=np.array(self.train)
            self.validation=np.array(self.validation)







    


    
