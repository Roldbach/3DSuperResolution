import numpy as np

from General.Configuration import experimentConfiguration
from General.DataWriting import saveData
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

def evaluatePrediction(prediction, target, name, path=experimentConfiguration.projectPath+"/Data"):
    '''
        Evaluate every prediction through RMSE, PSNR, SSIM with respect to the target
    in the 0~255 scale, save and report those results

        This function could be used for both 2D and 3D image arrays
        (1) For 2D image arrays, the input should be in the shape: (number, height, width)
        (2) For 3D image arrays, the input should be in the shape: (number, depth, height, width)
    '''
    rmse=[]
    psnr=[]
    ssim=[]

    for i in range(len(prediction)):
        rmse.append(superLoss((prediction[i]*255).astype("uint8"),(target[i]*255).astype("uint8"),mode="rmse"))      
        psnr.append(superLoss((prediction[i]*255).astype("uint8"),(target[i]*255).astype("uint8"),mode="psnr")) 
        ssim.append(superLoss((prediction[i]*255).astype("uint8"),(target[i]*255).astype("uint8"),mode="ssim")) 

    saveData(rmse, name+" rmse", path)
    saveData(psnr, name+" psnr", path)
    saveData(ssim, name+" ssim", path)

    print("The mean psnr is: ",np.mean(psnr))
    print("The std psnr is: ",np.std(psnr))
    print(" ")
    print("The mean ssim is: ",np.mean(ssim))
    print("The std ssim is: ",np.std(ssim))
    print(" ")
    print("The mean rmse is: ",np.mean(rmse))
    print("The std rmse is: ",np.std(rmse))
   