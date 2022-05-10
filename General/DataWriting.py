import nibabel as nib
import numpy as np
import torch

from General.Configuration import compressionConfiguration, experimentConfiguration, loadingConfiguration, optimizerConfiguration

def saveData(data, name, path=experimentConfiguration.projectPath+"/Data"):
    with open(path+"/"+name+".txt", "w") as file:
        for value in data:
            file.write(str(value)+"\n")

def saveModel(model, name, path=experimentConfiguration.projectPath+"/SavedModel", device_ids=None):
    if device_ids is None:
        torch.save(model.state_dict(), path+"/"+name+".pth")
    else:
        torch.save(model.module.state_dict(), path+"/"+name+".pth")

def saveLoss(trainLoss, validationLoss, name, path=experimentConfiguration.projectPath+"/Loss"):
    with open(path+"/"+name+".txt", "a") as file:
        file.write(f"train:{trainLoss},validation:{validationLoss}\n")

def save3DImage(image, name, path=experimentConfiguration.projectPath+"/Result"):
    affine = np.array([[0.,  0.,  1., -0.],
        [ 0., -1.,  0., -0.],
        [ -1.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  1.]])
    imageNil=nib.Nifti1Image(image, affine)
    nib.save(imageNil, path+"/"+name+".nii")

def saveSRExperiment(name, modelConfiguration, experimentConfiguration=experimentConfiguration, loadingConfiguration=loadingConfiguration,
                    compressionConfiguration=compressionConfiguration,optimizerConfiguration=optimizerConfiguration, path=experimentConfiguration.projectPath+"/SavedConfiguration"):
    with open(path+"/"+name+".txt", "w") as file:
        file.write(saveConfiguration(type(modelConfiguration).__name__[:-13], modelConfiguration))
        file.write(saveConfiguration(type(experimentConfiguration).__name__[:-13], experimentConfiguration))
        file.write(saveConfiguration(type(loadingConfiguration).__name__[:-13], loadingConfiguration))
        file.write(saveConfiguration(type(compressionConfiguration).__name__[:-13], compressionConfiguration))
        file.write(saveConfiguration(type(optimizerConfiguration).__name__[:-13], optimizerConfiguration))
    
def saveConfiguration(subName, configuration):
    ''' 
        Save the configuration in the format:
        Name: field 1,field 2,field 3....\n
    '''
    result=subName+":"
    for field in configuration._fields:
        result+=(field+"\t"+str(getattr(configuration,field))+"\t")
    return result+"\n"