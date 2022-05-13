import numpy as np

from General.Configuration import experimentConfiguration
from skimage.util import view_as_blocks, view_as_windows

def windowing(file,window=experimentConfiguration.window):
    '''
        Apply windowing to the image so the values are within the window range
    and normalized

        If no valid range for the window is given, the image is directly normalized

    input:
        file: dicom file, contain all information needed
        window: 2-D tuple, specify the window range
    
    return:
        image: np.ndarray, the modified image with all values within 0~1 as float32
    '''
    if window!=None:
        slope=file.RescaleSlope
        intercept=file.RescaleIntercept
        
        max=(window[1]-intercept)/slope
        min=(window[0]-intercept)/slope

        image=np.clip(file.pixel_array,min,max)
        image=(image-min)/(max-min)
        return image.astype("float32")
    
    else:
        image=file.pixel_array.astype("float32")      
        max=np.amax(image)
        min=np.amin(image)
        return (image-min)/(max-min)

def patchExtraction2D(dataset, windowShape=experimentConfiguration.patchWindowShape, step=experimentConfiguration.patchStep):
    result=[]
    for image in dataset:
        patch=view_as_windows(image, windowShape, step)
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                result.append(patch[i][j])
    return result

def patchExtraction3D(dataset, windowShape=experimentConfiguration.patchWindowShape):
    result=[]
    for i in range(len(dataset)):
        template=view_as_blocks(dataset[i], windowShape)
        for j in range(template.shape[0]):
            for k in range(template.shape[1]):
                for l in range(template.shape[2]):
                    result.append(template[j][k][l])
    return result
