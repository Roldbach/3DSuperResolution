import pydicom
from Handler.DataHandler import DataHandler
from Handler.ModelHandler import ModelHandler
from Handler.SRDataHandler import SRDataHandler
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as functional

from General.Configuration import *
from General.DataPlotting import plot3DImage, plotImage
from Model.DenseNet import DenseNet
from General.Evaluation import superLoss
from skimage.util import view_as_blocks
from Helper.TrainHelper import cleanResult, constructSetting
from torch.optim import Adam

unetConfiguration=UnetConfiguration(inputChannel=1, outputChannel=1, block=4, normalization=None, upMode='resizeconv_linear')
adamConfiguration=AdamConfiguration(rate=0.00001, beta=(0.9, 0.99))

test=ModelHandler(unetConfiguration, adamConfiguration)
device=torch.device("cuda")
device_ids=None
model, optimizer=test.constructModelOptimizerPair(device, device_ids)
print(type(model))
print(type(optimizer))
print(model)

'''
def phaseShift2D(input, factor):
    batchSize=input.shape[0]
    channel=input.shape[1]
    height=input.shape[2]
    width=input.shape[3]

    result=torch.reshape(input, (batchSize, height, width, factor, factor))
    #result=torch.transpose(input)
    print(result)
    print(result.shape)

layer_1=np.array([[2,1,9],[4,0,9],[3,0,0]])
layer_2=np.array([[4,3,0],[0,1,7],[2,8,4]])
layer_3=np.array([[5,8,9],[1,3,2],[0,1,6]])
layer_4=np.array([[0,4,1],[7,2,3],[4,5,3]])

test=np.stack([layer_1, layer_2, layer_3, layer_4])
test=np.expand_dims(test, axis=0)
test=torch.from_numpy(test)
phaseShift2D(test,2)
'''
'''

model=CustomizedPixelShuffle(2)

result=model(test)
print(result)
print(result.shape)
'''



