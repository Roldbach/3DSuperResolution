import time
import torch
import torch.nn as nn

from General.Configuration import experimentConfiguration, optimizerConfiguration, unetConfiguration
from General.DataLoading import loadPairPatch, loadLoss, loadModel
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

def trainBatch(noisy, clean, model, optimizer, loss, stepNumber, device, device_ids=None): 
    global trainLossTotal

    noisyBatch=getBatchData(noisy, stepNumber, len(noisy), device).unsqueeze(dim=1)
    cleanBatch=getBatchData(clean, stepNumber, len(clean), device).unsqueeze(dim=1)

    prediction=model(noisyBatch)

    optimizer.zero_grad()
    trainLossBatch=loss(prediction, cleanBatch)
    trainLossTotal+=trainLossBatch.item()
    trainLossBatch.backward()
    
    if device_ids!=None:
        optimizer.module.step()
    else:
        optimizer.step()

def validateBatch(noisy, clean, model, loss, stepNumber, device):
    global validationLossTotal

    noisyBatch=getBatchData(noisy, stepNumber, len(noisy), device).unsqueeze(dim=1)
    cleanBatch=getBatchData(clean, stepNumber, len(clean), device).unsqueeze(dim=1)
    
    prediction=model(noisyBatch)

    validationLossBatch=loss(prediction, cleanBatch)
    validationLossTotal+=validationLossBatch.item()

def predict(noisy, clean, model, name, saveResult=3):
    '''
        Return predictions and save results for the given number

        In order to predict the result using the full noisy input,
    this is done on CPU
    '''
    device=torch.device("cpu")
    model=model.to(device)
    result=[]

    with torch.no_grad():
        model.eval()
        for i in range(setting.testStep):
            noisyBatch=getSingleData(noisy, i, device).unsqueeze(dim=1)

            prediction=model(noisyBatch)
            prediction=prediction[0,0,:,:,:].detach().numpy()
            result.append(prediction)

            if i<saveResult:
                save3DImage(noisy[i], name+f" noisy {i}")
                save3DImage(clean[i], name+f" clean {i}")
                save3DImage(prediction[i], name+f" prediction {i}")

                plotImage(noisy[i][0], name+f" noisy front {i}")
                plotImage(clean[i][0], name+f" clean front {i}")
                plotImage(prediction[i][0], name+f" prediction front {i}")

                plotImage(noisy[i][:,:,100], name+f" noisy side {i}")
                plotImage(clean[i][:,:,100], name+f" clean side {i}")
                plotImage(prediction[i][:,:,100], name+f" prediction side {i}")
    
    return result
    
name="3D Unet full patch denoise"
loading=False

lowDose, fullDose=loadPairPatch()
setting=constructSetting(len(lowDose["train"]), len(lowDose["validation"]), len(lowDose["test"]))
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids=[0,1,2,3]
model, optimizer=constructModel(device, device_ids)
loss=nn.MSELoss().to(device)
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
        trainBatch(lowDose["train"], fullDose["train"], model, optimizer, loss, j, device, device_ids)
    trainLoss.append(trainLossTotal/setting.trainStep)
    
    with torch.no_grad():
        model.eval()
        for j in range(setting.validationStep):
            validateBatch(lowDose["validation"], fullDose["train"], model, loss, j, device)
        validationLoss.append(validationLossTotal/setting.validationStep)
    
    end=time.time()
    saveModel(model, name, device_ids=device_ids)
    saveLoss(trainLoss[-1], validationLoss[-1], name)
    print('Epoch : ',countEpoch(name), '\t', 'train loss: ',trainLoss[-1], '\t', "val loss: ", validationLoss[-1], 'time: ',end-start,"s")
    torch.cuda.empty_cache()

prediction=predict(lowDose["test"], fullDose["test"], model, name)
plotLoss(trainLoss, validationLoss, name)
evaluatePrediction(prediction, fullDose["test"], name)
#saveConfiguration(name, unetConfiguration)