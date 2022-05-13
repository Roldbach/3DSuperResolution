from General.Configuration import aapmMayoConfiguration, experimentConfiguration
from Handler.DataHandler import DataHandler

class SRDataHandler(DataHandler):
    '''
        The specific DataHandler class that could handle the data
    for super resolution tasks
    '''
    def __init__(self, experimentConfiguration=experimentConfiguration, loadingConfiguration=aapmMayoConfiguration):
        super().__init__(experimentConfiguration, loadingConfiguration)
    
    def trilinearInterpolation(self, LR, factor):
        '''
            According to the configuration, construct corresponding LR
        dataset from the HR dataset

        input:

        '''
