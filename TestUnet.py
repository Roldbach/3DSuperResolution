import numpy as np
import torch
import torch.nn as nn

from General.DataLoading import loadConfiguration, loadLoss, loadModel
from General.DataPlotting import plotImage, plotLoss
from General.DataWriting import savePerformance
from General.Evaluation import superLoss
from Handler.ExperimentHandler import ExperimentHandler
from Handler.ModelHandler import ModelHandler
from Handler.SRDataHandler import SRDataHandler

name="3DUnetPatch128"
resultPath="/home/weixun/3DSuperResolution/Result/"+name
configuration=loadConfiguration("Configuration", resultPath)
plotNumber=3

experimentHandler=ExperimentHandler(configuration["Experiment"])
dataHandler=SRDataHandler(configuration["Experiment"], configuration["Loading"], configuration["Downsample"], configuration["Upsample"])
modelHandler=ModelHandler(configuration["UNet"], configuration["Adam"])

local=torch.device("cpu")
device=experimentHandler.constructDevice()
device_ids=[0,1]

model=modelHandler.constructModel(device)
if device_ids!=None:
    model=nn.parallel.DataParallel(model, device_ids)

loadModel(model, "model", resultPath, device_ids)
trainLoss, validationLoss=loadLoss("loss", resultPath)
psnr, ssim, rmse=[], [], []

for i in range(len(dataHandler.testKey)):
    file, HR=dataHandler.getSingle(i)
    HRPatch=dataHandler.patchExtraction([HR])
    step=experimentHandler.calculateStep(HRPatch.shape[0], experimentHandler.experimentConfiguration.batchSize)

    LR=[]
    prediction=[]

    for j in range(step):
        HRPatchBatch=dataHandler.getBatch(HRPatch, j, device)
        LRPatchBatch=dataHandler.downsample(HRPatchBatch, device)
        LRPatchBatch=dataHandler.upsample(LRPatchBatch, device)

        predictionBatch=model(LRPatchBatch)

        LRPatchBatch=LRPatchBatch[:,0,:,:,:].cpu().detach().numpy()
        predictionBatch=predictionBatch[:,0,:,:,:].cpu().detach().numpy()
        LR+=[LRPatchBatch[k,:,:,:] for k in range(LRPatchBatch.shape[0])]
        prediction+=[predictionBatch[k,:,:,:] for k in range(predictionBatch.shape[0])]

    LR=dataHandler.patchReconstruction3D(LR, HR.shape)
    prediction=dataHandler.patchReconstruction3D(prediction, HR.shape)

    psnr.append(superLoss((255*prediction).astype("uint8"), (255*HR).astype("uint8"), mode="psnr"))
    ssim.append(superLoss((255*prediction).astype("uint8"), (255*HR).astype("uint8"), mode="ssim"))
    rmse.append(superLoss((255*prediction).astype("uint8"), (255*HR).astype("uint8"), mode="rmse"))

    if i<plotNumber:
        plotImage(LR[:,:,100], f"LR {i}", resultPath)
        plotImage(prediction[:,:,100], f"Prediction {i}", resultPath)
        plotImage(HR[:,:,100], f"HR {i}", resultPath)
    
    torch.cuda.empty_cache()

plotLoss(trainLoss, validationLoss, "loss plot", resultPath)

savePerformance(psnr, "psnr", resultPath)
savePerformance(ssim, "ssim", resultPath)
savePerformance(rmse, "rmse", resultPath)

print("The mean psnr is: ",np.mean(psnr))
print("The std psnr is: ",np.std(psnr))
print(" ")
print("The mean ssim is: ",np.mean(ssim))
print("The std ssim is: ",np.std(ssim))
print(" ")
print("The mean rmse is: ",np.mean(rmse))
print("The std rmse is: ",np.std(rmse))