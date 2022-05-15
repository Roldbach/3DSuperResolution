import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as functional
import matplotlib.pyplot as plt

from General.Configuration import compressionConfiguration, experimentConfiguration, optimizerConfiguration, unetConfiguration
from General.DataLoading import loadHRPatch, loadHRSlice, loadLoss, loadModel
from General.DataPlotting import plotImage, plotLoss
from General.DataWriting import saveConfiguration, saveLoss, saveModel, save3DImage
from General.Evaluation import evaluatePrediction
from torch.optim import Adam
from Train.TrainHelper import constructLoss, constructSetting, countEpoch, getBatchData, getSingleData
from Model.UNet import UNet

def constructModel(device, device_ids=None, inputChannel=unetConfiguration.inputChannel, outputChannel=unetConfiguration.outputChannel,
                    block=unetConfiguration.block, normalization=unetConfiguration.normalization, upMode=unetConfiguration.upMode):
    model=UNet(
        in_channels=inputChannel,
        out_channels=outputChannel,
        n_blocks=block,
        normalization=normalization,
        up_mode=upMode
    )
    model=model.to(device)
    optimizer=Adam(model.parameters(), optimizerConfiguration.learningRate, optimizerConfiguration.beta)
    
    if device_ids!=None:
        model=nn.DataParallel(model, device_ids)
        optimizer=nn.DataParallel(optimizer, device_ids)
        optimizer=optimizer.to(device)
    
    return model, optimizer

def trainBatch(HR, model, optimizer, loss, stepNumber, device, device_ids=None, factor=compressionConfiguration.factor,
    interpolation=compressionConfiguration.interpolation): 
    global trainLossTotal

    HRBatch=getBatchData(HR, stepNumber, len(HR), device).unsqueeze(dim=1)
    LRBatch=HRBatch[:,:,::factor,:,:]
    LRBatch=functional.interpolate(LRBatch,scale_factor=(factor,1,1), mode=interpolation)

    HRPredictiondiction=model(LRBatch)

    optimizer.zero_grad()
    trainLossBatch=loss(HRBatch, HRPredictiondiction)
    trainLossTotal+=trainLossBatch.item()
    trainLossBatch.backward()
    
    if device_ids!=None:
        optimizer.module.step()
    else:
        optimizer.step()

def validateBatch(HR, model, loss, stepNumber, device, factor=compressionConfiguration.factor, interpolation=compressionConfiguration.interpolation):
    global validationLossTotal

    HRBatch=getBatchData(HR, stepNumber, len(HR), device).unsqueeze(dim=1)
    LRBatch=HRBatch[:,:,::factor,:,:]
    LRBatch=functional.interpolate(LRBatch,scale_factor=(factor,1,1), mode=interpolation)
    
    HRPredictiontion=model(LRBatch)

    validationLossBatch=loss(HRBatch, HRPredictiontion)
    validationLossTotal+=validationLossBatch.item()

def predict(HR, HRSlice, model, device, name, saveResult=1, factor=compressionConfiguration.factor, interpolation=compressionConfiguration.interpolation):
    '''
        Return predictions and save results for the given number
    '''
    prediction=[]

    with torch.no_grad():
        model.eval()
        for i in range(setting.testStep):
            local=torch.device("cpu")
            HRBatch=getSingleData(HR, i, local).unsqueeze(dim=1)

            LRBatch=HRBatch[:,:,::factor,:,:]
            LRBatch=functional.interpolate(LRBatch, scale_factor=(factor,1,1), mode=interpolation)
            LRBatch=LRBatch.to(device)

            LRPrediction=model(LRBatch)
            LRPrediction=LRPrediction[0,0,:,:,:].cpu().detach().numpy()
            prediction.append(LRPrediction)

            if i<saveResult:
                LRBatch=LRBatch[0,0,:,:,:].cpu().detach().numpy()
                
                LRBatch=reconstructHU3D(LRBatch, HRSlice[i])
                HRImage=reconstructHU3D(HR[i], HRSlice[i])
                LRPrediction=reconstructHU3D(LRPrediction, HRSlice[i])

                save3DImage(LRBatch, name+f" LR {i}")
                save3DImage(HRImage, name+f" HR {i}")
                save3DImage(LRPrediction, name+f" Prediction {i}")
    
    return prediction

def reconstructHU3D(image, slice):
    for i in range(len(slice)):
        image[i]=reconstructHU(image[i], slice[i])
    return image

def reconstructHU(image, slice):
    '''
        Reconstruct HU using the head information in slice
    
    input:
        image: 2-D np.ndarray
        slice: dicom.dataframe, contains head information
    '''
    slope=slice.RescaleSlope
    intercept=slice.RescaleIntercept

    originalImage=slice.pixel_array
    max=np.amax(originalImage)
    min=np.amin(originalImage)

    reconstructImage=image*(max-min)+min
    return reconstructImage*slope+intercept

name="3D Unet full patch l4"
loading=True

HR=loadHRPatch()
HRSlice=loadHRSlice()

setting=constructSetting(len(HR["train"]), len(HR["validation"]), len(HR["test"]))
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids=[0,1]
model, optimizer=constructModel(device, device_ids)

if loading:
    loadModel(model, name, device_ids=device_ids)

prediction=predict(HR["test"], HRSlice["test"], model, device, name, 3)