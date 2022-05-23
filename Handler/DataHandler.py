import cv2
import numpy as np
import os
import pydicom as dicom
import torch

from General.ImageProcessing import patchExtraction2D, patchExtraction3D, patchReconstruction3D, windowing
from torch.autograd import Variable

class DataHandler:
    '''
        The class to handle various data loading and HU reconstruction
    '''
    def __init__(self, experimentConfiguration, loadingConfiguration):
        '''
            This class contains following attributes:
            (1) experimentConfiguration: namedtuple
            (2) loadingConfiguration: namedtuple
            (3) slice
                a. 2D: dict, key=file path, value=file path
                b. 3D: dict, key=patient name, value=tuple, contains all file paths
            (4) train/validation/test key: list, contains keys in those sub dataset
        '''
        self.experimentConfiguration=experimentConfiguration
        self.loadingConfiguration=loadingConfiguration

        self.slice=self.loadSlice()
        self.trainKey, self.validationKey, self.testKey=self.splitTrainValidationKey()
        
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
            result: dict, key=file path, value=file path
        '''
        result={}
        patientAll=tuple(sorted((self.experimentConfiguration.projectPath+"/"+"AapmMayo"+"/"+patient for patient in os.listdir(self.experimentConfiguration.projectPath+"/"+"AapmMayo")
                        if self.loadingConfiguration.resolution in patient)))
        
        for patient in patientAll:
            with open(patient, "r") as file:
                content=file.readlines()
            for line in content:
                result[line.strip("\n")]=line.strip("\n")
        
        return result
    
    def loadAapmMayoSlice3D(self):
        '''
            Load the dicom files from Aapm Mayo dataset with given resolution
        for training 3D model

            In this mode, files within the same folder are stored together
        
        return:
            result: dict, key=patient name, value=tuple, contains all file paths
        '''
        result={}
        patientAll=tuple(sorted((self.experimentConfiguration.projectPath+"/"+"AapmMayo"+"/"+patient for patient in os.listdir(self.experimentConfiguration.projectPath+"/"+"AapmMayo")
                if self.loadingConfiguration.resolution in patient)))
        
        for patient in patientAll:
            with open(patient, "r") as file:
                content=file.readlines()
            contentStrip=tuple(sorted((line.strip("\n") for line in content)))
            contentTruncate=self.truncateSlice(contentStrip, self.experimentConfiguration.patchWindowShape[0])
            result[patient]=contentTruncate
        
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
    
    def constructDataset(self, tag):
        '''
            Construct the corresponding dataset according to
        the tag and configuration
        '''
        if self.loadingConfiguration.mode=="2D":
            if tag=="train":
                return self.constructDataset2D(self.trainKey)
            elif tag=="validation":
                return self.constructDataset2D(self.validationKey)
            else:
                return self.constructDataset2D(self.testKey)
        else:
            if tag=="train":
                return self.constructDataset3D(self.trainKey)
            elif tag=="validation":
                return self.constructDataset3D(self.validationKey)
            else:
                return self.constructDataset3D(self.testKey)

    def constructDataset2D(self, keys):
        return [self.loadImage2D(key) for key in keys]
    
    def constructDataset3D(self, keys):
        return [self.loadImage3D(key) for key in keys]
    
    def loadImage2D(self, key):
        '''
            Load 2-D image with given key and apply
        windowing

        input:
            key: string, the file path of the dicom file

        return:
            result: np.ndarray, 2-D image after windowing
        '''
        result=self.loadFile2D(key)
        if self.loadingConfiguration.reshape!=None:
            return cv2.resize(windowing(result), self.loadingConfiguration.reshape)
        else:
            return windowing(result)
    
    def loadImage3D(self,key):
        '''
            Load 3-D image with given key and apply
        windowing

        input:
            key: string, the patient name

        return:
            result: np.ndarray, 3-D image after windowing
        '''
        file=self.loadFile3D(key)  
        if self.loadingConfiguration.reshape!=None:
            image=[cv2.resize(windowing(slice, self.experimentConfiguration.window), self.loadingConfiguration.reshape) for slice in file]
        else:
            image=[windowing(slice, self.experimentConfiguration.window) for slice in file]
        return np.stack(image).astype("float32")
    
    def loadFile2D(self, key):
        return dicom.read_file(self.slice[key])
    
    def loadFile3D(self, key):
        return [dicom.read_file(slice) for slice in self.slice[key]]

    def patchExtraction(self, dataset):
        '''
            Patch the dataset according to the experiment configuration

            If no valid patch window given, convert the dataset to np.array for training
        '''
        if len(self.experimentConfiguration.patchWindowShape)==3:
            return np.array(patchExtraction3D(dataset, self.experimentConfiguration.patchWindowShape))
        elif len(self.experimentConfiguration.patchWindowShape)==2:
            return np.array(patchExtraction2D(dataset, self.experimentConfiguration.patchWindowShape, self.experimentConfiguration.patchStep))
        else:
            return np.array(dataset)

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

    def getSingle(self, index):
        '''
            Return one image data from the test dataset
        with the given index
        '''
        if self.loadingConfiguration.mode=="2D":
            file=self.loadFile2D(self.testKey[index])
            image=windowing(file)
        else:
            file=self.loadFile3D(self.testKey[index])
            image=[windowing(slice, self.experimentConfiguration.window) for slice in file]
            image=np.stack(image).astype("float32")
            
        return file, image
    
    def imageToTensor(self, image, device):
        '''
            Convert the image to torch.tensor on the given device

            The image could be either 2D or 3D
        '''
        for i in range(2):
            image=np.expand_dims(image, axis=0)
        image=torch.from_numpy(image).float()
        return Variable(image, requires_grad=True).to(device)

    def patchReconstruction3D(self, patch, originalShape):
        '''
            Reconstruct a 3-D image to its original shape using
        all patches

            This function only works for non-overlapping patches
        (from view_as_blocks)
        '''
        return patchReconstruction3D(patch, originalShape, self.experimentConfiguration.patchWindowShape)

    


    
