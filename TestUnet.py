import torch

from General.DataLoading import loadConfiguration, loadLoss, loadModel
from General.DataPlotting import plotLoss
from Handler.SRDataHandler import SRDataHandler
from Model.UNet import UNet

name="3DUnetPatch128"
configuration=loadConfiguration()
resultPath=configuration["Experiment"].projectPath+"/"+name
plotNumber=3

HR=SRDataHandler(configuration["Experiment"], configuration["Loading"], configuration["Downsample"], configuration["Upsample"])
device=torch.device("cpu")
device_ids=None
trainLoss, validationLoss=loadLoss("loss", resultPath)
model=UNet(
    in_channels=configuration["Unet"].inputChannel,
    out_channels=configuration["Unet"].outputChannel,
    n_blocks=configuration["Unet"].block,
    normalization=configuration["Unet"].normalization,
    up_mode=configuration["Unet"].upMode
)
loadModel(model, "model", resultPath, device_ids)

result=[]
for i in range(len(HR.test)):
    HRBatch=HR.getSingle(HR.test, i, device)
    LRBatch=HR.downsample(HRBatch, device)
    LRBatch=HR.upsample(LRBatch, device)

    prediction=model(LRBatch)
    prediction=prediction[0,0,:,:,:].detach().numpy()
    result.append(prediction)

    if i<plotNumber:
        LRBatch=LRBatch[0,0,:,:,:].detach().numpy()
        



plotLoss(trainLoss, validationLoss, "loss plot", resultPath)
