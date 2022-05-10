import torch.nn as nn

from General.Configuration import intermediateCNNConfiguration

class IntermediateConvolution2DBlock(nn.Module):
    '''
        Standard 2D convolution layer + ReLU
    '''
    def __init__(self, inputChannel, outputChannel, kernel=intermediateCNNConfiguration.kernel, stride=intermediateCNNConfiguration.stride):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(inputChannel, outputChannel, kernel, stride, padding="same"),
            nn.ReLU()
        )
    
    def forward(self, input):
        return self.layer(input)

class IntermediateCNN(nn.Module):
    '''
        The network that could upsample height/width and 
    depth separately
    '''
    def __init__(self, mode, inputChannel=intermediateCNNConfiguration.inputChannel, channel=intermediateCNNConfiguration.channel, 
    factor=intermediateCNNConfiguration.factor):
        super(IntermediateCNN, self).__init__()
        self.mode=mode
        self.inputChannel=inputChannel
        self.channel=channel
        self.factor=factor

        self.before=self.constructBlock(intermediateCNNConfiguration.before)
        self.after=self.constructBlock(intermediateCNNConfiguration.after)
    
    def constructBlock(self, number):
        block=[]
        for i in range(number):
            if i==0:
                block.append(IntermediateConvolution2DBlock(self.inputChannel, self.channel))
            else:
                block.append(IntermediateConvolution2DBlock(self.channel, self.channel))
        return nn.ModuleList(block)