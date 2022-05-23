import torch
import torch.nn as nn

class DenseNetConvolution3DBlock(nn.Module):
    '''
        Standard 3D convolution layer + LeakyRelu + BatchNorm
    '''
    def __init__(self, inputChannel, outputChannel, kernel, stride):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv3d(inputChannel, outputChannel, kernel, stride, padding="same"),
            nn.LeakyReLU(),
            nn.BatchNorm3d(outputChannel)
        )
    
    def forward(self, input):
        return self.layer(input)

class DenseNet(nn.Module):
    '''
        Dense net backbone with standard 3D convolution
    '''
    def __init__(self, inputChannel, channel, level, kernel, stride):
        super(DenseNet, self).__init__()
        self.inputLayer=[DenseNetConvolution3DBlock(inputChannel, channel, kernel, stride)]
        self.intermediatelayer=[DenseNetConvolution3DBlock(i*channel, channel, kernel, stride) for i in range(1, level-1)]
        self.outputLayer=[DenseNetConvolution3DBlock((level-1)*channel, inputChannel, kernel, stride)]
        self.layer=nn.ModuleList(self.inputLayer+self.intermediatelayer+self.outputLayer)
    
    def forward(self, input):
        result=[]

        for i in range(len(self.layer)):
            if i==0:
                result.append(self.layer[i](input))
            else:
                inputConcat=torch.concat(result, dim=1)
                result.append(self.layer[i](inputConcat))  
        return result[-1]
