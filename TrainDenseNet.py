import time
import torch

from General.Configuration import AdamConfiguration, DownsampleConfiguration, ExperimentConfiguration, LoadingConfiguration, DenseNetConfiguration, UpsampleConfiguration
from General.DataLoading import loadLoss, loadModel
from General.DataWriting import saveDescription, saveExperiment, saveLoss, saveModel
from Handler.ExperimentHandler import ExperimentHandler
from Handler.ModelHandler import ModelHandler
from Handler.SRDataHandler import SRDataHandler

def trainBatch(train, dataHandler, model, optimizer, loss, step, device, device_ids=None): 
    global trainLossTotal

    HRBatch=dataHandler.getBatch(train, step, device)
    LRBatch=dataHandler.downsample(HRBatch, device)
    LRBatch=dataHandler.upsample(LRBatch, device)

    HRPrediction=model(LRBatch)

    optimizer.zero_grad()
    trainLossBatch=loss(HRBatch, HRPrediction)
    trainLossTotal+=trainLossBatch.item()
    trainLossBatch.backward()
    
    if device_ids!=None:
        optimizer.module.step()
    else:
        optimizer.step()

def validateBatch(validation, dataHandler, model, loss, step, device):
    global validationLossTotal

    HRBatch=dataHandler.getBatch(validation, step, device)
    LRBatch=dataHandler.downsample(HRBatch, device)
    LRBatch=dataHandler.upsample(LRBatch, device)
    
    HRPrediction=model(LRBatch)

    validationLossBatch=loss(HRBatch, HRPrediction)
    validationLossTotal+=validationLossBatch.item()

#-----Configuration-----#
experimentConfiguration=ExperimentConfiguration(
    #projectPath="/content/drive/MyDrive/Code/3DSuperResolution",
    name="DenseNet",
    mode="train",
    projectPath="/home/weixun/3DSuperResolution",
    window=None, patchWindowShape=(64,64,64), patchStep=None,
    batchSize=32, epoch=500, loss="L1"
)
aapmMayoConfiguration=LoadingConfiguration(path="/media/NAS01/Aapm-Mayo/LDCT-and-Projection-data",
                      #path="/content/drive/MyDrive/Code/3DSuperResolution/3D-MNIST",
                      mode="3D", resolution="HR", resample=False, trainProportion=0.8, validationProportion=0.1)
downsampleConfiguration=DownsampleConfiguration(name="interval", factor=8)

upsampleConfiguration=UpsampleConfiguration(name="trilinear interpolation", factor=(8,1,1))

denseNetConfiguration=DenseNetConfiguration(inputChannel=1, channel=64, level=6, kernel=3, stride=1)

adamConfiguration=AdamConfiguration(rate=0.00001, beta=(0.9, 0.99))
#-----------------------#

#-----Description-----#
description=[
    "This experiment is to test the performance of the DenseNet with standard 3D convolution."
]
#---------------------#

device_ids=None
experimentHandler=ExperimentHandler(experimentConfiguration)
modelHandler=ModelHandler(denseNetConfiguration, adamConfiguration)
dataHandler=SRDataHandler(experimentConfiguration, aapmMayoConfiguration, downsampleConfiguration, upsampleConfiguration)

device=experimentHandler.constructDevice()
train, validation=dataHandler.constructDataset("train"), dataHandler.constructDataset("validation")
train, validation=dataHandler.patchExtraction(train), dataHandler.patchExtraction(validation)
setting=experimentHandler.constructSetting(train.shape[0], validation.shape[0])
loss=experimentHandler.constructLoss(device)
resultPath=experimentHandler.constructResultPath()
model, optimizer=modelHandler.constructModelOptimizerPair(device, device_ids)
trainLoss, validationLoss=[],[]

if experimentConfiguration.mode=="load":
    loadModel(model, "model", resultPath, device_ids)
    trainLoss, validationLoss=loadLoss("loss", resultPath)
else:
    experimentHandler.clean(resultPath)
    resultPath=experimentHandler.constructResultPath()

for i in range(experimentConfiguration.epoch):
    start=time.time()
    trainLossTotal=0
    validationLossTotal=0

    model.train()
    for j in range(setting.trainStep):
        trainBatch(train, dataHandler, model, optimizer, loss, j, device, device_ids)
    trainLoss.append(trainLossTotal/setting.trainStep)
    
    with torch.no_grad():
        model.eval()
        for j in range(setting.validationStep):
            validateBatch(validation, dataHandler, model, loss, j, device)
        validationLoss.append(validationLossTotal/setting.validationStep)
    
    end=time.time()
    saveModel(model, "model", resultPath, device_ids)
    saveLoss(trainLoss[-1], validationLoss[-1], "loss", resultPath)
    print('Epoch : ',experimentHandler.countEpoch("loss", resultPath), '\t', 'train loss: ',trainLoss[-1], '\t', "val loss: ", validationLoss[-1], 'time: ',end-start,"s")
    torch.cuda.empty_cache()

saveDescription(description, "Description", resultPath)
saveExperiment("Configuration", resultPath, experimentConfiguration, aapmMayoConfiguration, downsampleConfiguration, upsampleConfiguration, denseNetConfiguration, adamConfiguration)