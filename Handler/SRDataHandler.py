import torch.nn.functional as functional

from Handler.DataHandler import DataHandler

class SRDataHandler(DataHandler):
    '''
        The specific DataHandler class that could handle the data
    for super resolution tasks
    '''
    def __init__(self, experimentConfiguration, loadingConfiguration, downsampleConfiguration, upsampleConfiguration):
        super().__init__(experimentConfiguration, loadingConfiguration)
        self.downsampleConfiguration=downsampleConfiguration
        self.upsampleConfiguration=upsampleConfiguration

    def downsample(self, data, device):
        '''
            Downsample the 3D image according to the configuration
        '''
        if self.downsampleConfiguration.name=="interval":
            return self.intervalDownsample(data, self.downsampleConfiguration.factor, device)

    def intervalDownsample(self, data, factor, device):
        '''
            Downsample the 3D image only in z-axis by taking out constant
        intervals

        This function only accepts 5D torch.tensor as input
        
        input:
            data: torch.tensor, this could only be in the shape: (number, channel, depth, height, width)
            factor: int, the interval to extract slices
            device: torch.device, either cpu or gpu
        '''
        return data[:, :, ::factor, :, :].to(device)

    def upsample(self, data, device):
        '''
            Upsample the 3D image according to the upsample configuration

            The following method could be chosen:
            (1) trilinear interpolation: this could only work for 5D torch.tensor
        '''
        if self.upsampleConfiguration.name=="trilinear interpolation":
            return self.trilinearInterpolation(data, self.upsampleConfiguration.factor, device)

    def trilinearInterpolation(self, data, factor, device):
        '''
            Use trilinear interpolation to upsample the given data to the given factor

            This function only accepts 5D torch.tensor as input
        
        input:
            data: torch.tensor, this could only be in the shape: (number, channel, depth, height, width)
            factor: tuple, the upsample factor in the shape: (depth factor, height factor, width factor)
            device: torch.device, either cpu or gpu
        '''
        return functional.interpolate(data, scale_factor=factor, mode="trilinear").to(device)


