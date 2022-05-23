import os
import torch
import torch.nn as nn
import shutil

from collections import namedtuple

class ExperimentHandler:
    '''
        The class to handle important operations in the experiment
    '''
    def __init__(self, experimentConfiguration):
        self.experimentConfiguration=experimentConfiguration
    
    def constructDevice(self):
        '''
            If gpu is available, use gpu!
        '''
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def constructSetting(self, trainSize, validationSize):
        '''
            Calculate the number of steps required for each epoch
        during train and validation
        '''
        Setting=namedtuple("Setting", ["trainStep", "validationStep"])

        setting=Setting(
            trainStep=self.calculateStep(trainSize, self.experimentConfiguration.batchSize),
            validationStep=self.calculateStep(validationSize, self.experimentConfiguration.batchSize)
        )
        return setting

    def calculateStep(self, size, batchSize):
        '''
            Calculate the steps required with the given length
        and batch size
        '''
        if size%batchSize!=0:
            step=size//batchSize+1
        else:
            step=size//batchSize
        return step
    
    def constructLoss(self, device):
        '''
            Return the loss function according to the 
        configuration and move it to the given device
        '''
        if self.experimentConfiguration.loss=="L2":
            return nn.MSELoss().to(device)
        elif self.experimentConfiguration.loss=="L1":
            return nn.L1Loss().to(device)
    
    def constructResultPath(self):
        '''
            Create a new directory for saving results
        from a new experiment and return the path
        '''
        try:
            os.mkdir(self.experimentConfiguration.projectPath+"/Result/"+self.experimentConfiguration.name)
            return self.experimentConfiguration.projectPath+"/Result/"+self.experimentConfiguration.name
        except:
            return self.experimentConfiguration.projectPath+"/Result/"+self.experimentConfiguration.name
    
    def countEpoch(self, name, path):
        '''
            Return the current epoch number based on the
        loss data in the file
        '''
        try:
            with open(path+"/"+name+".txt", "r") as file:
                lines=file.readlines()
            return len(lines)
        except FileNotFoundError:
            print("The loss file was not successfully loaded. Please try again.")
    
    def clean(self, path):
        '''
            If not in loading mode but the directory
        already exists, this must be the unwanted result
        from the last experiment and should be cleaned
        '''
        if os.path.isdir(path):
            shutil.rmtree(path)