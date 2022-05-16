import nibabel as nib
import numpy as np
import torch

def savePerformance(data, name, path):
    with open(path+"/"+name+".txt", "w") as file:
        for value in data:
            file.write(str(value)+"\n")

def saveModel(model, name, path, device_ids=None):
    if device_ids is None:
        torch.save(model.state_dict(), path+"/"+name+".pth")
    else:
        torch.save(model.module.state_dict(), path+"/"+name+".pth")

def saveLoss(trainLoss, validationLoss, name, path):
    with open(path+"/"+name+".txt", "a") as file:
        file.write(f"train:{trainLoss},validation:{validationLoss}\n")

def save3DImage(image, name, path):
    affine = np.array([[0.,  0.,  1., -0.],
        [ 0., -1.,  0., -0.],
        [ -1.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  1.]])
    imageNil=nib.Nifti1Image(image, affine)
    nib.save(imageNil, path+"/"+name+".nii")

def saveExperiment(name,path, *configuration):
    with open(path+"/"+name+".txt", "w") as file:
        for item in configuration:
            file.write(saveConfiguration(type(item).__name__[:-13], item))
    
def saveConfiguration(subName, configuration):
    ''' 
        Save the configuration in the format:
        Name: field 1,field 2,field 3....\n
    '''
    result=subName+":"
    for field in configuration._fields:
        result+=(field+"\t"+str(getattr(configuration,field))+"\t")
    return result+"\n"