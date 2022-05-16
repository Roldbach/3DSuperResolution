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
    def __init__(self, inputChannel, channel, level):
        super(DenseNet, self).__init__()
        self.channel=channel
        self.level=level

        self.layer=[DenseNetConvolution3DBlock(inputChannel)]+[DenseNetConvolution3DBlock(i*self.channel) for i in range(1,self.level-1)]+[DenseNetConvolution3DBlock((self.level-1)*self.channel,inputChannel)]
        self.layer=nn.ModuleList(self.layer)
    
    def forward(self, input):
        result=[]

        for i in range(self.level):
            if i==0:
                result.append(self.layer[i](input))
            else:
                inputConcat=torch.concat(result, dim=1)
                result.append(self.layer[i](inputConcat))  
        return result[-1]
