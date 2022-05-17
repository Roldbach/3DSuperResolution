import time
import torch
import torch.nn as nn

from General.Configuration import AdamConfiguration, DownsampleConfiguration, ExperimentConfiguration, LoadingConfiguration, UnetConfiguration, UpsampleConfiguration
from General.DataLoading import loadLoss, loadModel
from General.DataWriting import saveDescription, saveExperiment, saveLoss, saveModel
from Handler.ExperimentHandler import ExperimentHandler
from Handler.ModelHandler import ModelHandler
from Handler.SRDataHandler import SRDataHandler
from Helper.TrainHelper import cleanResult, countEpoch

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

#-----Configuration-----#
experimentConfiguration=ExperimentConfiguration(
    #projectPath="/content/drive/MyDrive/Code/3DSuperResolution",
    name="3DUnetPatch128",
    mode="load",
    device_ids=[0,1,2,3],
    projectPath="/home/weixun/3DSuperResolution",
    window=None, patchWindowShape=(64,128,128), patchStep=None,
    batchSize=32, epoch=500, loss="L2"
)
aapmMayoConfiguration=LoadingConfiguration(path="/media/NAS01/Aapm-Mayo/LDCT-and-Projection-data",
                      #path="/content/drive/MyDrive/Code/3DSuperResolution/3D-MNIST",
                      mode="3D", resolution="HR", resample=False, trainProportion=0.8, validationProportion=0.1)
downsampleConfiguration=DownsampleConfiguration(name="interval", factor=8)

upsampleConfiguration=UpsampleConfiguration(name="trilinear interpolation", factor=(8,1,1))

unetConfiguration=UnetConfiguration(inputChannel=1, outputChannel=1, block=4, normalization=None, upMode='resizeconv_linear')

adamConfiguration=AdamConfiguration(rate=0.00001, beta=(0.9, 0.99))
#-----------------------#

#-----Description-----#
description=[
    "This experiment is to test the performance of increasing the patch size to (64,128,128)."
]
#---------------------#

experimentHandler=ExperimentHandler(experimentConfiguration)
modelHandler=ModelHandler(unetConfiguration, adamConfiguration)
HR=SRDataHandler(experimentConfiguration, aapmMayoConfiguration, downsampleConfiguration, upsampleConfiguration)

device=experimentHandler.constructDevice()
setting=experimentHandler.constructSetting(HR.train.shape[0], HR.validation.shape[0], len(HR.test))
loss=experimentHandler.constructLoss(device)
resultPath=experimentHandler.constructResultPath(experimentConfiguration.name, experimentConfiguration.projectPath)
model, optimizer=modelHandler.constructModelOptimizerPair(device, experimentConfiguration.device_ids)
trainLoss, validationLoss=[],[]

if experimentConfiguration.mode=="load":
    loadModel(model, "model", resultPath, device_ids=experimentConfiguration.device_ids)
    trainLoss, validationLoss=loadLoss("loss", resultPath)
else:
    cleanResult(resultPath)

for i in range(experimentConfiguration.epoch):
    start=time.time()
    trainLossTotal=0
    validationLossTotal=0

    model.train()
    for j in range(setting.trainStep):
        trainBatch(HR, model, optimizer, loss, j, device, experimentConfiguration.device_ids)
    trainLoss.append(trainLossTotal/setting.trainStep)
    
    with torch.no_grad():
        model.eval()
        for j in range(setting.validationStep):
            validateBatch(HR, model, loss, j, device)
        validationLoss.append(validationLossTotal/setting.validationStep)
    
    end=time.time()
    saveModel(model, "model", resultPath, device_ids=experimentConfiguration.device_ids)
    saveLoss(trainLoss[-1], validationLoss[-1], "loss", resultPath)
    print('Epoch : ',countEpoch("loss", resultPath), '\t', 'train loss: ',trainLoss[-1], '\t', "val loss: ", validationLoss[-1], 'time: ',end-start,"s")
    torch.cuda.empty_cache()

saveDescription(description, "Description", resultPath)
saveExperiment("Configuration", resultPath, experimentConfiguration, aapmMayoConfiguration, downsampleConfiguration, upsampleConfiguration, unetConfiguration, adamConfiguration)

