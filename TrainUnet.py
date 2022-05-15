import time

import torch
import torch.nn as nn
import torch.nn.functional as functional

from General.Configuration import compressionConfiguration, experimentConfiguration, optimizerConfiguration, unetConfiguration
from General.DataLoading import loadHRPatch, loadLoss, loadModel
from General.DataPlotting import plotImage, plotLoss
from General.DataWriting import saveSRExperiment, saveLoss, saveModel, save3DImage
from General.Evaluation import evaluatePrediction
from Handler.SRDataHandler import SRDataHandler
from torch.optim import Adam
from TrainHelper import constructLoss, constructSetting, countEpoch, getBatchData, getSingleData
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

def trainBatch(HR, model, optimizer, loss, step, device, device_ids=None): 
    global trainLossTotal

    HRBatch=HR.getBatch(HR.train, step, device)
    LRBatch=HR.downsample(HRBatch, device)
    LRBatch=HR.upsample(LRBatch, device)

    HRPrediction=model(LRBatch)

    optimizer.zero_grad()
    trainLossBatch=loss(HRBatch, HRPrediction)
    trainLossTotal+=trainLossBatch.item()
    trainLossBatch.backward()
    
    if device_ids!=None:
        optimizer.module.step()
    else:
        optimizer.step()

def validateBatch(HR, model, loss, step, device):
    global validationLossTotal

    HRBatch=HR.getBatch(HR.train, step, device)
    LRBatch=HR.downsample(HRBatch, device)
    LRBatch=HR.upsample(LRBatch, device)
    
    HRPrediction=model(LRBatch)

    validationLossBatch=loss(HRBatch, HRPrediction)
    validationLossTotal+=validationLossBatch.item()

def predict(HR, model, device, name, saveResult=1, factor=compressionConfiguration.factor, interpolation=compressionConfiguration.interpolation):
    '''
        Return predictions and save results for the given number
    '''
    prediction=[]

    with torch.no_grad():
        model.eval()
        for i in range(setting.testStep):
            HRBatch=getSingleData(HR, i, device).unsqueeze(dim=1)
            LRBatch=HRBatch[:,:,::factor,:,:]
            LRBatch=functional.interpolate(LRBatch, scale_factor=(factor,1,1), mode=interpolation)

            LRPrediction=model(LRBatch)
            LRPrediction=LRPrediction[0,0,:,:,:].cpu().detach().numpy()
            prediction.append(LRPrediction)

            if i<saveResult:
                LRBatch=LRBatch[0,0,:,:,:].cpu().detach().numpy()
                save3DImage(LRBatch, name+f" LR {i}")
                save3DImage(HR[i], name+f" HR {i}")
                save3DImage(prediction[i], name+f" Prediction {i}")

                plotImage(LRBatch[:,:,100], name+f" LR {i}")
                plotImage(HR[i][:,:,100], name+f" HR {i}")
                plotImage(prediction[i][:,:,100], name+f" Prediction {i}")
    
    return prediction

loading=False

HR=SRDataHandler()
setting=constructSetting(HR.train.shape[0], HR.validation.shape[0], len(HR.test))
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids=None
model, optimizer=constructModel(device, device_ids)
loss=nn.MSELoss().to(device)
trainLoss, validationLoss=constructLoss()

if loading:
    loadModel(model, experimentConfiguration.name, device_ids=device_ids)
    trainLoss, validationLoss=loadLoss(experimentConfiguration.name)

for i in range(experimentConfiguration.epoch):
    start=time.time()
    trainLossTotal=0
    validationLossTotal=0

    model.train()
    for j in range(setting.trainStep):
        trainBatch(HR, model, optimizer, loss, j, device, device_ids)
    trainLoss.append(trainLossTotal/setting.trainStep)
    
    with torch.no_grad():
        model.eval()
        for j in range(setting.validationStep):
            validateBatch(HR, model, loss, j, device)
        validationLoss.append(validationLossTotal/setting.validationStep)
    
    end=time.time()
    saveModel(model, experimentConfiguration.name, device_ids=device_ids)
    saveLoss(trainLoss[-1], validationLoss[-1], experimentConfiguration.name)
    print('Epoch : ',countEpoch(experimentConfiguration.name), '\t', 'train loss: ',trainLoss[-1], '\t', "val loss: ", validationLoss[-1], 'time: ',end-start,"s")
    torch.cuda.empty_cache()

    break

#prediction=predict(HR["test"], model, device, name, saveResult=1)
#plotLoss(trainLoss, validationLoss, name)
#evaluatePrediction(prediction, HR["test"], name)
#saveSRExperiment(name, unetConfiguration)

