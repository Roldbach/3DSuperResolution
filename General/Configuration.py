from collections import namedtuple

ExperimentConfiguration=namedtuple("ExperimentConfiguration",["name", "projectPath", "window", "patchWindowShape", "patchStep", "batchSize", "epoch"])
LoadingConfiguration=namedtuple("LoadingConfiguration", ["path", "mode", "resolution", "resample", "trainProportion", "validationProportion"])
DownsampleConfiguration=namedtuple("DownsampleConfiguration", ["name", "factor"])
UpsampleConfiguration=namedtuple("UpsampleConfiguration", ["name", "factor"])
UnetConfiguration=namedtuple("UnetConfiguration", ["inputChannel", "outputChannel", "block", "normalization", "upMode"])
QueueBlockConfiguration=namedtuple("QueueBlockConfiguration", ["intermediateChannel", "depthKernel", "heightKernel", "widthKernel", "stride"])
ParallelNetConfiguration=namedtuple("ParallelNetConfiguration", ["inputChannel", "factor", "channel", "level", "kernel", "stride"])
IntermediateCNNConfiguration=namedtuple("IntermediateConfiguration", ["inputChannel", "factor", "channel", "kernel", "stride", "before", "after"])
DenseNetConfiguration=namedtuple("DenseNetConfiguration",["inputChannel", "factor", "channel", "level", "kernel", "stride"])
AdamConfiguration=namedtuple("AdamConfiguration", ["rate", "beta"])

#Queue Block Configuration
queueBlockConfiguration=QueueBlockConfiguration(intermediateChannel=32, depthKernel=3, heightKernel=3, widthKernel=3, stride=1)

#ParallelNet Model Configuration
parallelNetConfiguration=ParallelNetConfiguration(inputChannel=1, channel=32, level=5, factor=2, kernel=3, stride=1)

#Intermediate CNN Model Configuration
intermediateCNNConfiguration=IntermediateCNNConfiguration(inputChannel=1, channel=32, factor=4, kernel=3, stride=1, before=6, after=3)

#Dense Net Model Configuration
denseNetConfiguraiton=DenseNetConfiguration(inputChannel=1, factor=8, channel=64, level=6, kernel=3, stride=1)
