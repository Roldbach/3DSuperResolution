import time
import torch
import torch.nn as nn

from General.Configuration import AdamConfiguration, DownsampleConfiguration, ExperimentConfiguration, LoadingConfiguration, UnetConfiguration, UpsampleConfiguration
from General.DataLoading import loadLoss, loadModel
from General.DataWriting import saveExperiment, saveLoss, saveModel
from Handler.SRDataHandler import SRDataHandler
from torch.optim import Adam
from Helper.TrainHelper import constructLoss, constructPath, constructSetting, countEpoch
from Model.UNet import UNet

def constructModel(unetConfiguration, optimizerConfiguration, device, device_ids=None):
    model=UNet(
        in_channels=unetConfiguration.inputChannel,
        out_channels=unetConfiguration.outputChannel,
        n_blocks=unetConfiguration.block,
        normalization=unetConfiguration.normalization,
        up_mode=unetConfiguration.upMode
    )
    model=model.to(device)
    optimizer=Adam(model.parameters(), optimizerConfiguration.rate, optimizerConfiguration.beta)
    
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

#-----Configuration-----#
experimentConfiguration=ExperimentConfiguration(
    #projectPath="/content/drive/MyDrive/Code/3DSuperResolution",
    name="3DUnetPatch",
    projectPath="/home/weixun/3DSuperResolution",
    window=None, patchWindowShape=(64,64,64), patchStep=None, batchSize=16, epoch=500
)
aapmMayoConfiguration=LoadingConfiguration(path="/media/NAS01/Aapm-Mayo/LDCT-and-Projection-data",
                      #path="/content/drive/MyDrive/Code/3DSuperResolution/3D-MNIST",
                      mode="3D", resolution="HR", resample=False, trainProportion=0.8, validationProportion=0.1)
downsampleConfiguration=DownsampleConfiguration(name="interval", factor=8)

upsampleConfiguration=UpsampleConfiguration(name="trilinear interpolation", factor=(8,1,1))

unetConfiguration=UnetConfiguration(inputChannel=1, outputChannel=1, block=4, normalization=None, upMode='resizeconv_linear')

adamConfiguration=AdamConfiguration(rate=0.00001, beta=(0.9, 0.99))
#-----------------------#

loading=False
HR=SRDataHandler(experimentConfiguration, aapmMayoConfiguration, downsampleConfiguration, upsampleConfiguration)
setting=constructSetting(HR.train.shape[0], HR.validation.shape[0], len(HR.test), experimentConfiguration.batchSize)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids=[0,1,2,3]
model, optimizer=constructModel(unetConfiguration, adamConfiguration, device, device_ids)
loss=nn.MSELoss().to(device)
trainLoss, validationLoss=constructLoss()
savePath=constructPath(experimentConfiguration.name, experimentConfiguration.projectPath)

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
    saveModel(model, "model", savePath, device_ids=device_ids)
    saveLoss(trainLoss[-1], validationLoss[-1], "loss", savePath)
    print('Epoch : ',countEpoch("loss", savePath), '\t', 'train loss: ',trainLoss[-1], '\t', "val loss: ", validationLoss[-1], 'time: ',end-start,"s")
    torch.cuda.empty_cache()

saveExperiment("Configuration", savePath, experimentConfiguration, aapmMayoConfiguration, downsampleConfiguration, upsampleConfiguration, unetConfiguration, adamConfiguration)

