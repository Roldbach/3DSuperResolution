import torch
import torch.nn as nn

from Model.UNet import UNet
from torch.optim import Adam

class ModelHandler:
    '''
        The class to construct different models and optimizers
    with given configurations
    '''
    def __init__(self, modelConfiguration, optimizerConfiguration):
        '''
            This class contains following attributes:
            (1) modelConfiguration: namedtuple
            (2) optimizerConfiguration: namedtuple
        '''
        self.modelConfiguration=modelConfiguration
        self.optimizerConfiguration=optimizerConfiguration
    
    def constructModelOptimizerPair(self, device, device_ids=None):
        '''
            Return the model and the corresponding optimizer with given
        configurations
        '''
        model=self.constructModel(device)
        optimizer=self.constructOptimizer(model.parameters())

        if device_ids!=None and device==torch.device("cuda"):
            model=nn.parallel.DataParallel(model, device_ids)
            optimizer=nn.parallel.DataParallel(optimizer, device_ids)
            optimizer=optimizer.to(device)
        
        return model, optimizer

    def constructModel(self, device):
        '''
            Return the model to the given device

            If multiple device ids are given, return
        the model in parallel mode
        '''
        if "Unet" in type(self.modelConfiguration).__name__:
            return self.constructUNetModel(device)
    
    def constructUNetModel(self, device):
        model=UNet(
            in_channels=self.modelConfiguration.inputChannel,
            out_channels=self.modelConfiguration.outputChannel,
            n_blocks=self.modelConfiguration.block,
            normalization=self.modelConfiguration.normalization,
            up_mode=self.modelConfiguration.upMode
        )
        return model.to(device)
    
    def constructOptimizer(self, modelParameter):
        '''
            Return the optimizer with given configuration

            To return the optimizer in parallel mode, multiple
        gpu have to be available
        '''
        if "Adam" in type(self.optimizerConfiguration).__name__:
            return self.constructAdamOptimizer(modelParameter)
    
    def constructAdamOptimizer(self, modelParameter):
        return Adam(modelParameter, self.optimizerConfiguration.rate, self.optimizerConfiguration.beta)
