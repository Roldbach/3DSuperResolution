import time
import torch
import torch.nn as nn
import torch.nn.functional as functional

from General.Configuration import experimentConfiguration, optimizerConfiguration, parallelNetConfiguration
from General.DataLoading import loadHRPatch, loadLoss, loadModel
from General.DataPlotting import plotImage, plotLoss
from General.DataWriting import saveSRExperiment, saveLoss, saveModel, save3DImage
from General.Evaluation import evaluatePrediction
from torch.optim import Adam
from Model.ParallelNet import ParallelNet
from Train.TrainHelper import constructLoss, constructSetting, countEpoch, getBatchData, getSingleData

def constructModel(device, device_ids=None):
    model=ParallelNet()
    model=model.to(device)
    optimizer=Adam(model.parameters(), optimizerConfiguration.learningRate, optimizerConfiguration.beta)
    
    if device_ids!=None:
        model=nn.DataParallel(model, device_ids)
        optimizer=nn.DataParallel(optimizer, device_ids)
        optimizer=optimizer.to(device)
    
    return model, optimizer

def trainBatch(HR, model, optimizer, loss, stepNumber, device, device_ids=None): 
    global trainLossTotal

    HRBatch=getBatchData(HR, stepNumber, len(HR), device).unsqueeze(dim=1)
    LRBatch=functional.interpolate(HRBatch,scale_factor=(0.5,0.5,0.5), mode="trilinear")

    HRPrediction=model(LRBatch)

    optimizer.zero_grad()
    trainLossBatch=loss(HRBatch, HRPrediction)
    trainLossTotal+=trainLossBatch.item()
    trainLossBatch.backward()
    
    if device_ids!=None:
        optimizer.module.step()
    else:
        optimizer.step()

def validateBatch(HR, model, loss, stepNumber, device):
    global validationLossTotal

    HRBatch=getBatchData(HR, stepNumber, len(HR), device).unsqueeze(dim=1)
    LRBatch=functional.interpolate(HRBatch,scale_factor=(0.5,0.5,0.5), mode="trilinear")
    
    HRPrediction=model(LRBatch)

    validationLossBatch=loss(HRBatch, HRPrediction)
    validationLossTotal+=validationLossBatch.item()

def predict(HR, model, device, name, saveResult=3):
    '''
        Return predictions and save results for the given number
    '''
    prediction=[]

    with torch.no_grad():
        model.eval()
        for i in range(setting.testStep):
            HRBatch=getSingleData(HR, i, device).unsqueeze(dim=1)
            LRBatch=functional.interpolate(HRBatch,scale_factor=(0.5,0.5,0.5), mode="trilinear")

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
    
name="ParallelNet"
loading=False

HR=loadHRPatch()
setting=constructSetting(len(HR["train"]), len(HR["validation"]), len(HR["test"]))
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids=[0,1,2,3]
model, optimizer=constructModel(device, device_ids)
loss=nn.L1Loss().to(device)
trainLoss, validationLoss=constructLoss()

if loading:
    loadModel(model, name, device_ids=device_ids)
    trainLoss, validationLoss=loadLoss(name)

for i in range(experimentConfiguration.epoch):
    start=time.time()
    trainLossTotal=0
    validationLossTotal=0

    model.train()
    for j in range(setting.trainStep):
        trainBatch(HR["train"], model, optimizer, loss, j, device, device_ids)
    trainLoss.append(trainLossTotal/setting.trainStep)
    
    with torch.no_grad():
        model.eval()
        for j in range(setting.validationStep):
            validateBatch(HR["validation"], model, loss, j, device)
        validationLoss.append(validationLossTotal/setting.validationStep)
    
    end=time.time()
    saveModel(model, name, device_ids=device_ids)
    saveLoss(trainLoss[-1], validationLoss[-1], name)
    print('Epoch : ',countEpoch(name), '\t', 'train loss: ',trainLoss[-1], '\t', "val loss: ", validationLoss[-1], 'time: ',end-start,"s")
    torch.cuda.empty_cache()

prediction=predict(HR["test"], model, device, name)
plotLoss(trainLoss, validationLoss, name)
evaluatePrediction(prediction, HR["test"], name)
#saveSRExperiment(name, parallelNetConfiguration)

    
    




