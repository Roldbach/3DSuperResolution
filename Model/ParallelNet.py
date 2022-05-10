import numpy as np
import torch
import torch.nn as nn

from General.Configuration import  parallelNetConfiguration,queueBlockConfiguration

class CustomizedPixelShuffle(nn.Module):
    '''
        Modify the pixel shuffle function in the source
    code so it could be used for N-D data
    '''
    def __init__(self, upscaleFactor):
        super(CustomizedPixelShuffle, self).__init__()
        self.upscaleFactor=upscaleFactor
    
    def forward(self, input):
        return self.pixel_shuffle(input, self.upscaleFactor)
    
    def pixel_shuffle(self, input, upscale_factor):
        '''
            Modify the pixel shuffle function in the source
        code so it could be used for N-D data
        '''
        input_size = list(input.size())
        dimensionality = len(input_size) - 2

        input_size[1] //= (upscale_factor ** dimensionality)
        output_size = [dim * upscale_factor for dim in input_size[2:]]

        input_view = input.contiguous().view(
            input_size[0], input_size[1],
            *(([upscale_factor] * dimensionality) + input_size[2:])
        )

        indicies = list(range(2, 2 + 2 * dimensionality))
        shuffle_out = input_view.permute(0, 1, *(indicies[::-1])).contiguous()
        return shuffle_out.view(input_size[0], input_size[1], *output_size)

class ParallelNetConvolution3DBlock(nn.Module):
    '''
        Standard 3D convolution layer + ReLU
    '''
    def __init__(self, inputChannel, outputChannel, kernel=parallelNetConfiguration.kernel, stride=parallelNetConfiguration.stride):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv3d(inputChannel, outputChannel, kernel, stride, padding="same"),
            nn.ReLU()
        )
    
    def forward(self, input):
        return self.layer(input)

class QueueBlock(nn.Module):
    '''
        Use standard 1-D convolution across all channels in 3 axis to
    replace a standard 3-D convolution
    '''
    def __init__(self, inputChannel, outputChannel,
                intermediateChannel=queueBlockConfiguration.intermediateChannel,
                depthKernel=queueBlockConfiguration.depthKernel,
                heightKernel=queueBlockConfiguration.heightKernel,
                widthKernel=queueBlockConfiguration.widthKernel, stride=queueBlockConfiguration.stride):
        super(QueueBlock, self).__init__()
        self.depthKernel=(depthKernel,1,1)
        self.heightKernel=(1,heightKernel,1)
        self.widthKernel=(1,1,widthKernel)
        self.stride=stride
    
        self.layer=nn.Sequential(
            nn.Conv3d(inputChannel, intermediateChannel, (1,1,1), 1, "same"),
            nn.Conv3d(intermediateChannel, intermediateChannel, self.depthKernel, self.stride, "same"),
            nn.Conv3d(intermediateChannel, intermediateChannel, self.heightKernel, self.stride, "same"),
            nn.Conv3d(intermediateChannel, intermediateChannel, self.widthKernel, self.stride, "same"),
            nn.Conv3d(intermediateChannel, outputChannel, (1,1,1), 1, "same")
        )

    def forward(self, input):
        return self.layer(input)

class ParallelNet(nn.Module):
    '''
        Parallel net backbone with standard 3D convolution layer
    '''
    def __init__(self, inputChannel=parallelNetConfiguration.inputChannel, channel=parallelNetConfiguration.channel,
    level=parallelNetConfiguration.level, factor=parallelNetConfiguration.factor):
        super(ParallelNet, self).__init__()
        self.inputChannel=inputChannel
        self.channel=channel
        self.level=level
        self.factor=factor
        self.inputBlock=ParallelNetConvolution3DBlock(self.inputChannel, np.power(self.factor, 3))
        
        self.block={}
        for i in range(self.level):
            for j in range(1, self.level+1-i):
                if (i+1,j)==(1,1):
                    self.block[f"{i+1} {j}"]=ParallelNetConvolution3DBlock(8, self.channel*(i+1))
                else:
                    self.block[f"{i+1} {j}"]=ParallelNetConvolution3DBlock(self.channel*(i+1), self.channel*(i+1))
        for i in range(2,self.level+1):
            self.block[f"{i} {0}"]=ParallelNetConvolution3DBlock(self.channel*sum([j for j in range(i)]), self.channel*i)
        self.block=nn.ModuleDict(self.block)

        self.outputLayer=nn.Conv3d(self.channel*sum([i for i in range(1,self.level+1)]), np.power(self.factor, 3), parallelNetConfiguration.kernel, parallelNetConfiguration.stride, "same")
        self.shuffle=CustomizedPixelShuffle(self.factor)
    
    def forward(self, input):
        result={}
        for i in range(1, self.level+1):
            for j in range(self.level+2-i):
                if (i,j)==(1,0):
                    result[f"{i} {j}"]=self.inputBlock(input)
                elif j==0:
                    layer=[(k, i-k) for k in range(1, i)]
                    concatenatation=[result[f"{element[0]} {element[1]}"] for element in layer]
                    result[f"{i} {j}"]=self.block[f"{i} {j}"](torch.concat(concatenatation, dim=1))
                else:
                    result[f"{i} {j}"]=self.block[f"{i} {j}"](result[f"{i} {j-1}"])
        
        layer=[(i, self.level+1-i) for i in range(1, self.level+1)]
        concatenatation=[result[f"{element[0]} {element[1]}"] for element in layer]
        output=self.outputLayer(torch.concat(concatenatation, dim=1))
        output+=result["1 0"]
        return self.shuffle(output)