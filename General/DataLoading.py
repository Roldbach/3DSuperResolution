import numpy as np
import torch

from scipy import ndimage

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

def loadLoss(name, path):
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

def loadModel(model, name, path, device_ids):
    if device_ids!=None:
        model.module.load_state_dict(torch.load(path+"/"+name+".pth"))
    else:
        model.load_state_dict(torch.load(path+"/"+name+".pth"))