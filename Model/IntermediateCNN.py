import torch.nn as nn

class IntermediateConvolution2DBlock(nn.Module):
    '''
        Standard 2D convolution layer + ReLU
    '''
    def __init__(self, inputChannel, outputChannel, kernel, stride):
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
    def __init__(self, mode, inputChannel, channel, factor, before, after):
        super(IntermediateCNN, self).__init__()
        self.mode=mode
        self.inputChannel=inputChannel
        self.channel=channel
        self.factor=factor

        self.before=self.constructBlock(before)
        self.after=self.constructBlock(after)
    
    def constructBlock(self, number):
        block=[]
        for i in range(number):
            if i==0:
                block.append(IntermediateConvolution2DBlock(self.inputChannel, self.channel))
            else:
                block.append(IntermediateConvolution2DBlock(self.channel, self.channel))
        return nn.ModuleList(block)