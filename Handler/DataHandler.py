import numpy as np
import os
import pydicom as dicom
import torch

from General.ImageProcessing import patchExtraction2D, patchExtraction3D, windowing
from torch.autograd import Variable

class DataHandler:
    '''
        The class to handle various data loading and HU reconstruction
    '''
    def __init__(self, experimentConfiguration, loadingConfiguration):
        '''
            This class contains following attributes:
            (1) slice
                a. 2D: dict, key=file path, value=corresonding dicom file
                b. 3D: dict, key=patient name, value=tuple, contains all files
            (2) image
                a. 2D: dict, key=file path, value=window/normalized 2D image
                b. 3D: dict, key=patinet name, value=window/normalized 3D image
            (3) train/validation/test key: list, contains keys in those sub dataset
            (4) train/validation/test: list, contains 2D/3D images in those sub dataset
            (5) setting: namedtuple, contains the number of step required in one epoch for all sub dataset
        '''
        self.experimentConfiguration=experimentConfiguration
        self.loadingConfiguration=loadingConfiguration

        self.slice=self.loadSlice()
        self.image=self.loadImage()
        self.trainKey, self.validationKey, self.testKey=self.splitTrainValidationKey()
        self.train, self.validation, self.test=self.splitTrainValidationImage()
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
    
        #Unfinished Function
    def generateOSICPath(self,datasetPath):
        image=[]

        patient=self.filterPath(datasetPath, ".7z")
        for path in patient:
            patientID=self.filterPath(datasetPath+"/"+path, ".csv", ".txt")
            imagePath=[datasetPath+"/"+path+"/"+element+"/0" for element in patientID]
            imagePath.sort()
            for element in imagePath:
                break

    def filterPath(self, path, *target):
        '''
        Load files from the given path and filter out
        files whose name contains the target
        '''
        result=sorted(os.listdir(path))
        for item in target:
            result=[element for element in result if item not in element]
        return result

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
            contentTruncate=self.truncateSlice(contentStrip, self.experimentConfiguration.patchWindowShape[0])
            result[patient]=[dicom.read_file(slice) for slice in contentTruncate]
        
        return result

    def truncateSlice(self, slice, depth):
        '''
            Truncate the slice in z-axis to allow non-overlapping voxel patch
        extraction

            If the window is not given, simply return all slices
        '''
        try:
            truncateTotal=len(slice)%depth
            truncateStart=int(truncateTotal/2)
            truncateEnd=truncateTotal-truncateStart
            return slice[truncateStart:len(slice)-truncateEnd]
        except:
            return slice

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
            image=[windowing(file, self.experimentConfiguration.window) for file in value]
            result[key]=np.stack(image).astype("float32")
        return result
    
    def splitTrainValidationKey(self):
        '''
            Split the slice dataset into train, validation and test dataset
        according to the given proportions
        '''
        keys=list(self.slice.keys())

        trainLimit=int(len(keys)*self.loadingConfiguration.trainProportion)
        validationLimit=int(len(keys)*self.loadingConfiguration.validationProportion)
        
        trainKey=keys[len(keys)-trainLimit:len(keys)]
        validationKey=keys[len(keys)-trainLimit-validationLimit:len(keys)-trainLimit]
        testKey=keys[:len(keys)-trainLimit-validationLimit]

        return trainKey, validationKey, testKey
    
    def splitTrainValidationImage(self):
        '''
            Split the image dataset into train, validation and test dataset
        according to the given proportions
        '''
        train=[self.image[key] for key in self.trainKey]
        validation=[self.image[key] for key in self.validationKey]
        test=[self.image[key] for key in self.testKey]

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

    def getBatch(self, dataset, step, device):
        '''
            Return a batch of data from dataset 
        and transfer to the given device

        input:
            dataset: list, contains all data used
            step: int, the current number of experiment step
            device: torch.device, either cpu or gpu

        return:
            result: torch.Variable, could be directly used for train/validation
        '''
        if (step+1)*self.experimentConfiguration.batchSize<=len(dataset):
            result=dataset[step*self.experimentConfiguration.batchSize:(step+1)*self.experimentConfiguration.batchSize]
        else:
            result=dataset[step*self.experimentConfiguration.batchSize:len(dataset)]
        result=np.expand_dims(result, axis=1)
        result=torch.from_numpy(result).float()
        return Variable(result, requires_grad=True).to(device)
    
    def getSingle(self, dataset, index, device):
        '''
            Return one single data from dataset and
        transfer to the given device

        input:
            dataset: list, contains all data used
            index: int, the index of the target
            device: torch.device, either cpu or gpu
        
        return:
            result: torch.Variable, could be directly used for test
        '''
        result=dataset[index]
        for i in range(2):
            result=np.expand_dims(result, axis=0)

        result=torch.from_numpy(result).float()
        return Variable(result, requires_grad=True).to(device)


    


    
