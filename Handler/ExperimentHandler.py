import os
import torch
import torch.nn as nn

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
    
    def constructSetting(self, trainSize, validationSize, testSize):
        '''
            Calculate the number of steps required for each epoch
        during training, validating and testing

            For testing, the prediction is generated 1 by 1
        '''
        Setting=namedtuple("Setting", ["trainStep", "validationStep", "testStep"])

        setting=Setting(
            trainStep=self.calculateStep(trainSize, self.experimentConfiguration.batchSize),
            validationStep=self.calculateStep(validationSize, self.experimentConfiguration.batchSize),
            testStep=self.calculateStep(testSize, 1)
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