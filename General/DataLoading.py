import cv2
import numpy as np
import os
import pydicom as dicom
import torch

from General.Configuration import compressionConfiguration, experimentConfiguration, loadingConfiguration
from scipy import ndimage

#Unfinished Function
def loadOSIC(datasetPath):
		image=[]
		
		patient=filterPath(datasetPath, ".7z")
		for path in patient:
				patientID=filterPath(datasetPath+"/"+path, ".csv", ".txt")
				imagePath=[datasetPath+"/"+path+"/"+element+"/0" for element in patientID]
				imagePath.sort()
				for element in imagePath:
						image.append(loadImageFromSlice(element))
				break

def filterPath(path, *target):
    '''
        Load files from the given path and filter out
    files whose name contains the target
    '''
    result=sorted(os.listdir(path))
    for item in target:
        result=[element for element in result if item not in element]
    return result

def loadPairPatch(path=experimentConfiguration.datasetPath, windowShape=experimentConfiguration.patchWindowShape,
    trainProportion=loadingConfiguration.trainProportion, validationProportion=loadingConfiguration.validationProportion):
    '''
		Load LDCT+FDCT from Aapm dataset and store it as train, validation, test data
	in 0~1 scale according to the proportion given

        The patch extraction could only be applied to the train and validation
    dataset if given patch window shape

        Each sub dataset is in the shape: (number, depth, height, width)
    '''
    lowDose, fullDose=loadAapmMayo(path)
    lowDose=splitTrainValidation(lowDose, trainProportion, validationProportion)
    fullDose=splitTrainValidation(fullDose, trainProportion, validationProportion)

    if windowShape!=None:
        lowDose["train"]=patchExtraction3D(lowDose["train"], windowShape)
        lowDose["validation"]=patchExtraction3D(lowDose["validation"], windowShape)

        fullDose["train"]=patchExtraction3D(fullDose["train"], windowShape)
        fullDose["validation"]=patchExtraction3D(fullDose["validation"], windowShape)

    return lowDose, fullDose

def loadAapmMayo(datasetPath=experimentConfiguration.datasetPath):
    '''
        Load all Aapm Mayo dataset
    '''
    patientList = os.listdir(datasetPath)
    lowDose = []
    fullDose = []

    for patient in patientList:
        patientSub=os.listdir(datasetPath+"/"+patient)
        for fileSub in patientSub:
            imgPath=datasetPath+"/"+patient+"/"+fileSub
            if "full" in fileSub.lower():
                fullDose.append(loadImageFromSlice(imgPath))
            else:
                lowDose.append(loadImageFromSlice(imgPath))

    return lowDose, fullDose

def loadHRPatch(path=experimentConfiguration.datasetPath, windowShape=experimentConfiguration.patchWindowShape,
    trainProportion=loadingConfiguration.trainProportion, validationProportion=loadingConfiguration.validationProportion):
    '''
		Load FDCT from Aapm dataset and store it as train, validation, test data
	in 0~1 scale according to the proportion given

        The patch extraction could only be applied to the train and validation
    dataset if given patch window shape

        Each sub dataset is in the shape: (number, depth, height, width)
    '''
    HR=loadAapmMayoHR(path)
    HR=splitTrainValidation(HR, trainProportion, validationProportion)

    if windowShape!=None:
        HR["train"]=patchExtraction3D(HR["train"], windowShape)
        HR["validation"]=patchExtraction3D(HR["validation"], windowShape)

    return HR

def loadAapmMayoHR(datasetPath=experimentConfiguration.datasetPath):
    '''
    	Load the Aapm Mayo dataset and only return FDCT images
    '''
    patientList = os.listdir(datasetPath)
    fullDose = []

    for patient in patientList:
        patientSub=os.listdir(datasetPath+"/"+patient)
        for fileSub in patientSub:
            imgPath=datasetPath+"/"+patient+"/"+fileSub
            if "full" in fileSub.lower():
                fullDose.append(loadImageFromSlice(imgPath))
    return fullDose

def loadAapmMayoHRTest(datasetPath=experimentConfiguration.datasetPath):
    '''
    	Load the Aapm Mayo dataset and only return FDCT images
    '''
    patientList = os.listdir(datasetPath)
    fullDose = []

    for patient in patientList:
        patientSub=os.listdir(datasetPath+"/"+patient)
        for fileSub in patientSub:
            imgPath=datasetPath+"/"+patient+"/"+fileSub
            if "full" in fileSub.lower():
                fullDose.append(imgPath)
    return sorted(fullDose)

def loadImageFromSlice(datasetPath):
    slice=loadScan(datasetPath)
    image=loadImage(slice)
    image=truncate3DImage(image)
    return image

def loadScan(datasetPath):
    '''
        Load all slices for a certain patient using slice position
    '''
    slice=[dicom.read_file(datasetPath+"/"+fileName) for fileName in os.listdir(datasetPath)]
    slice.sort(key=lambda dicomFile: int(dicomFile.InstanceNumber))

    try:
        sliceThickness=np.abs(slice[0].ImagePositionPatient[2]-slice[1].ImagePositionPatient[2])
    except:
        sliceThickness=np.abs(slice[0].SliceLocation-slice[1].SliceLocation)

    for element in slice:
        element.sliceThickness=sliceThickness

    return slice

def loadImage(slice, size=compressionConfiguration.size):
    '''
        Load and stack 3D image arrays according to
    all dicom files for a certain patient and rescale
    to 0~1

        If the size is given in:
        (1) 2D: resize each 2D image array into this dimension
        (2) 3D: only store the first part of the 3D image arrays
    '''
    if len(size)==2:
        image=np.stack([cv2.resize(element.pixel_array, (size[0],size[1])) for element in slice]).astype("float32")
    elif len(size)==3:
        image=np.stack([cv2.resize(element.pixel_array, (size[1],size[2])) for element in slice]).astype("float32")
        image=image[:size[0]]
    else:
        image=np.stack(slice).astype("float32")

    max=np.amax(image)
    min=np.amin(image)
    image=(image-min)/(max-min)

    return image

def resample(image, slice, newSpacing=[1,1,1]):
	'''
        Resample the 3D image array so they have
    the same z-axis resolution
    '''
    # Determine current pixel spacing
	spacing = map(float, ([slice[0].SliceThickness] + list(slice[0].PixelSpacing)))
	spacing = np.array(list(spacing))

	resizeFactor = spacing / newSpacing
	newShape = image.shape * resizeFactor
	newShape = np.round(newShape)
	realFactor = newShape / image.shape
	newSpacing = spacing / realFactor
	
	image = ndimage.interpolation.zoom(image, realFactor)
	
	return image, newSpacing

def splitTrainValidation(dataset, trainProportion=loadingConfiguration.trainProportion, validationProportion=loadingConfiguration.validationProportion):
    '''
        Split the given dataset into train, validation and test dataset
    according to the given proportions

    input:
        dataset: List, contains all images in one place
        trainProportion: float, the proportion of training data
        validationProportion: float, the proportion of validation data
    
    return:
        result: dictionary, contains following key-value pairs:
        (1) train: list, contains all images in the training dataset
        (2) validation: list, contains all images in the validation dataset
        (3) test: list, contains all images in the testing dataset
    '''
    trainLimit=int(len(dataset)*trainProportion)
    validationLimit=int(len(dataset)*validationProportion)

    training=dataset[len(dataset)-trainLimit:len(dataset)]
    validation=dataset[len(dataset)-trainLimit-validationLimit:len(dataset)-trainLimit]
    test=dataset[:len(dataset)-trainLimit-validationLimit]

    return {"train":training, "validation":validation, "test":test}

def truncate3DImage(image, depth=experimentConfiguration.patchWindowShape[0]):
    '''
        Truncate the 3D image arrays in z-axis to allow fixed window
    patch extraction

    input:
        image: 3D np.ndarray, in the shape (depth, height, width)
        depth: Integer, the depth of the patch window
    '''
    truncateTotal=image.shape[0]%depth
    truncateStart=int(truncateTotal/2)
    truncateEnd=truncateTotal-truncateStart
    return image[truncateStart:image.shape[0]-truncateEnd]

def loadLoss(name, path=experimentConfiguration.projectPath+"/Loss"):
    try:
        trainLoss=[]
        validationLoss=[]
        
        with open(path+"/"+name+".txt", "r") as file:
            lines=file.readlines()
        
        for line in lines:
            content=line.strip("\n").split(",")
            for element in content:
                index=element.index(":")
                if "train"==element[:index]:
                    trainLoss.append(float(element[index+1:]))
                else:
                    validationLoss.append(float(element[index+1:]))
        
        return trainLoss, validationLoss
        
    except:
        print("Can't open loss text file.")

def loadModel(model, name, path=experimentConfiguration.projectPath+"/SavedModel", device_ids=None):
    if device_ids!=None:
        model.module.load_state_dict(torch.load(path+"/"+name+".pth"))
    else:
        model.load_state_dict(torch.load(path+"/"+name+".pth"))

#Uncertain Part
def loadHRSlice(path=experimentConfiguration.datasetPath,trainProportion=loadingConfiguration.trainProportion,
                validationProportion=loadingConfiguration.validationProportion):
    '''
        Load FDCT slices from Aapm dataset and store it as train, validation, test data
    '''
    HRSlice=loadAapmMayoSlice(path)
    HRSlice=splitTrainValidation(HRSlice, trainProportion, validationProportion)

    return HRSlice

def loadAapmMayoSlice(datasetPath=experimentConfiguration.datasetPath):
    '''
        Load the Aapm Mayo dataset and only return FDCT dicom files
    '''
    patientList = os.listdir(datasetPath)
    fullDoseSlice = []

    for patient in patientList:
        patientSub=os.listdir(datasetPath+"/"+patient)
        for fileSub in patientSub:
            imgPath=datasetPath+"/"+patient+"/"+fileSub
            if "full" in fileSub.lower():
                fullDoseSlice.append(loadSlice(imgPath))
    return fullDoseSlice

def loadSlice(datasetPath):
    slice=loadScan(datasetPath)
    slice=truncateSlice(slice)
    return slice

def truncateSlice(slice, depth=experimentConfiguration.patchWindowShape[0]):
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