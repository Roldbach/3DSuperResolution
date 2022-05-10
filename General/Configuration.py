from collections import namedtuple

ExperimentConfiguration=namedtuple("ExperimentConfiguration",["datasetPath", "projectPath", "window",
                        "patchWindowShape", "patchStep", "batchSize", "epoch"])
LoadingConfiguration=namedtuple("LoadingConfiguration", ["path", "mode", "resolution", "trainProportion", "validationProportion"])
UnetConfiguration=namedtuple("UnetConfiguration", ["inputChannel", "outputChannel", "block", "normalization", "upMode"])
QueueBlockConfiguration=namedtuple("QueueBlockConfiguration", ["intermediateChannel", "depthKernel", "heightKernel", "widthKernel", "stride"])
ParallelNetConfiguration=namedtuple("ParallelNetConfiguration", ["inputChannel", "factor", "channel", "level", "kernel", "stride"])
IntermediateCNNConfiguration=namedtuple("IntermediateConfiguration", ["inputChannel", "factor", "channel", "kernel", "stride", "before", "after"])
DenseNetConfiguration=namedtuple("DenseNetConfiguration",["inputChannel", "factor", "channel", "level", "kernel", "stride"])
OptimizerConfiguration=namedtuple("OptimizerConfiguration", ["learningRate", "beta"])
CompressionConfiguration=namedtuple("CompressionConfiguration", ["size", "factor", "interpolation"])


#General Experiment Configuration
experimentConfiguration=ExperimentConfiguration(
    #datasetPath="/content/drive/MyDrive/Code/3DSuperResolution/3D-MNIST",
    #projectPath="/content/drive/MyDrive/Code/3DSuperResolution",
    datasetPath="/media/NAS01/Aapm-Mayo/LDCT-and-Projection-data",
    projectPath="/home/weixun/3DSuperResolution",
    window=None, patchWindowShape=(64,64,64), patchStep=16, batchSize=32, epoch=500
)

loadingConfiguration=LoadingConfiguration(path="/media/NAS01/Aapm-Mayo/LDCT-and-Projection-data",
                      #path="/content/drive/MyDrive/Code/3DSuperResolution/3D-MNIST",
                      mode="3D", resolution="HR", trainProportion=0.8, validationProportion=0.1)



#Aapm Mayo Dataset Loading Configuration
aapmMayoConfiguration=LoadingConfiguration(path="/media/NAS01/Aapm-Mayo/LDCT-and-Projection-data",
                      #path="/content/drive/MyDrive/Code/3DSuperResolution/3D-MNIST",
                      mode="3D", resolution="HR", trainProportion=0.8, validationProportion=0.1)

#Unet Model Configuration
unetConfiguration=UnetConfiguration(inputChannel=1, outputChannel=1, block=4, normalization=None, upMode='resizeconv_linear')

#Queue Block Configuration
queueBlockConfiguration=QueueBlockConfiguration(intermediateChannel=32, depthKernel=3, heightKernel=3, widthKernel=3, stride=1)

#ParallelNet Model Configuration
parallelNetConfiguration=ParallelNetConfiguration(inputChannel=1, channel=32, level=5, factor=2, kernel=3, stride=1)

#Intermediate CNN Model Configuration
intermediateCNNConfiguration=IntermediateCNNConfiguration(inputChannel=1, channel=32, factor=4, kernel=3, stride=1, before=6, after=3)

#Dense Net Model Configuration
denseNetConfiguraiton=DenseNetConfiguration(inputChannel=1, factor=8, channel=64, level=6, kernel=3, stride=1)

#General Optimizer Configuration
optimizerConfiguration=OptimizerConfiguration(learningRate=0.00001, beta=(0.9,0.99))

#Compression Configuration
compressionConfiguration=CompressionConfiguration(size=(256,256), factor=8, interpolation="trilinear")
