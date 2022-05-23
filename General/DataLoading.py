import numpy as np
import torch

from General.Configuration import AdamConfiguration, DownsampleConfiguration, ExperimentConfiguration, LoadingConfiguration, UNetConfiguration, UpsampleConfiguration 
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

def loadModel(model, name, path, device_ids=None):
    if device_ids!=None:
        model.module.load_state_dict(torch.load(path+"/"+name+".pth"))
    else:
        model.load_state_dict(torch.load(path+"/"+name+".pth"))

def loadConfiguration(name, path):
    '''
        Load the target configuration text file and 
    return all stored configuraitons in order

    input:
        path: string, the path to the target text file
    '''
    with open(path+"/"+name+".txt", "r") as file:
        lines=file.readlines()
    
    result={}
    for line in lines:
        name, content=constructContent(line)
        result[name]=constructConfiguration(name, content)
    
    return result

def constructConfiguration(name, content):
    '''
        Return the corresponding configuration according to the 
    given name and content

    input:
        name: string, the name of the target configuration
        content: dict, contains all information with corresponding keys
    '''
    if name=="Experiment":
        return ExperimentConfiguration(name=content["name"], mode=content["mode"], projectPath=content["projectPath"],
                                    window=content["window"], patchWindowShape=content["patchWindowShape"], patchStep=content["patchStep"],
                                    batchSize=content["batchSize"], epoch=content["epoch"], loss=content["loss"])
    elif name=="Loading":
        return LoadingConfiguration(path=content["path"], mode=content["mode"], resolution=content["resolution"],
                                    resample=content["resample"], trainProportion=content["trainProportion"], validationProportion=content["validationProportion"])
    elif name=="Downsample":
        return DownsampleConfiguration(name=content["name"], factor=content["factor"])
    elif name=="Upsample":
        return UpsampleConfiguration(name=content["name"], factor=content["factor"])
    elif name=="UNet":
        return UNetConfiguration(inputChannel=content["inputChannel"], outputChannel=content["outputChannel"], block=content["block"],
                                normalization=content["normalization"], upMode=content["upMode"])
    elif name=="Adam":
        return AdamConfiguration(rate=content["rate"], beta=content["beta"])            

def constructContent(line):
    '''
        Return the name and a dictionary of parameters which could be used
    to constuct a configuration

    input:
        line: string, the line directly read from the text file
    '''
    content=line.strip("\n")
    index=content.index(":")
    name=content[:index]
    content=content[index+1:].split("\t")
    content.pop()

    result={}
    for i in range(0,len(content)-1,2):
        result[content[i]]=convertType(content[i+1])
    
    return name, result

def convertType(content):
    '''
        Return the content in the correct type

        This function could handle the following situations:
        1. If there is a "." in the content, it must be a float
        2. If it could be converted to an int, it must be an int
        3. If "(" or ")" within the content, it must be a tuple and the content
            of the tuple will be further checked
        4. If the content is "True" or "False", it must be a boolean
        5. If the content is "None", it must be a None
        6. If not above, it must be a string
    
    input:
        content: string, the content stored in the text file
    '''
    if "(" in content:
        return convertTuple(content)
    elif "." in content:
        return float(content)
    elif "e-" in content:
        return float(content)
    elif "None"==content:
        return None
    elif "True"==content:
        return True
    elif "False"==content:
        return False

    try:
        return int(content)
    except:
        return content

def convertTuple(content):
    '''
        Return the content as a tuple and every
    item wihtin the tuple is also converted to
    the right type 
    '''
    return tuple((convertType(item) for item in content[1:-1].split(",")))