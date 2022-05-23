import numpy as np

from General.DataWriting import savePerformance
from math import sqrt
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

def superLoss(prediction, target, mode="psnr",data_range=255):
    '''
        Calculate various loss between given 2 images and return the result
    
        This function works for both 2D and 3D images
    '''
    if mode=="rmse":
        return sqrt(MSE(prediction,target))
    elif mode=="psnr":
        return PSNR(prediction,target,data_range=data_range)
    else:
        return SSIM(prediction,target,data_range=data_range)