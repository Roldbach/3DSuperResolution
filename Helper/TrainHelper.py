import os
import numpy as np
import torch

from collections import namedtuple
from torch.autograd import Variable

def constructLoss():
    trainLoss=[]
    validationLoss=[]
    return trainLoss, validationLoss

def constructPath(name, projectPath):
    '''
        Create a new directory for saving results
    from a new experiment and return the path

    input:
        name: string, the name of the experiment
        projectPath: string, the root path of the project
    
    return:
        result: string, the path to save results for this experiment
    '''
    try:
        os.mkdir(projectPath+"/Result/"+name)
        return projectPath+"/Result/"+name
    except:
        return projectPath+"/Result/"+name

def constructSetting(trainLength, validationLength, testLength, batchSize):
    '''
        Calculate the number of steps required for each epoch
    during training, validating and testing

        For testing, the prediction is generated 1 by 1
    '''
    Setting=namedtuple("Setting", ["trainStep", "validationStep", "testStep"])

    setting=Setting(
        trainStep=calculateStep(trainLength, batchSize),
        validationStep=calculateStep(validationLength, batchSize),
        testStep=calculateStep(testLength, 1)
    )
    return setting

def calculateStep(length, batchSize):
    if length%batchSize!=0:
        step=length//batchSize+1
    else:
        step=length//batchSize
    return step

def getBatchData(dataset, stepNumber, length, device, batchSize):
    if (stepNumber+1)*batchSize<=length:
        result=dataset[stepNumber*batchSize:(stepNumber+1)*batchSize]
    else:
        result=dataset[stepNumber*batchSize:length]
    return arrayToVariable(result, device)

def getSingleData(dataset, index, device):
    result=dataset[index:index+1]
    return arrayToVariable(result,device)

def arrayToVariable(input, device):
    if not isinstance(input, np.ndarray):
        input=np.array(input)
    tensor=torch.from_numpy(input).float()
    return Variable(tensor, requires_grad=True).to(device)

def countEpoch(name, path):
    '''
        Return the current epoch number based on the
    loss data in the file
    '''
    try:
        with open(path+"/"+name+".txt", "r") as file:
            lines=file.readlines()
        return len(lines)
    except FileNotFoundError:
        print("The file was not successfully loaded. Please try again.")