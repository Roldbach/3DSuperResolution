from collections import namedtuple

ExperimentConfiguration=namedtuple("ExperimentConfiguration",["name", "mode", "projectPath", "window", "patchWindowShape", "patchStep", "batchSize", "epoch", "loss"])
LoadingConfiguration=namedtuple("LoadingConfiguration", ["path", "mode", "resolution", "resample", "reshape", "trainProportion", "validationProportion"])
DownsampleConfiguration=namedtuple("DownsampleConfiguration", ["name", "factor"])
UpsampleConfiguration=namedtuple("UpsampleConfiguration", ["name", "factor"])
UNetConfiguration=namedtuple("UNetConfiguration", ["inputChannel", "outputChannel", "block", "normalization", "upMode"])
QueueBlockConfiguration=namedtuple("QueueBlockConfiguration", ["intermediateChannel", "depthKernel", "heightKernel", "widthKernel", "stride"])
ParallelNetConfiguration=namedtuple("ParallelNetConfiguration", ["inputChannel", "channel", "level", "factor", "kernel", "stride"])
IntermediateCNNConfiguration=namedtuple("IntermediateConfiguration", ["inputChannel", "factor", "channel", "kernel", "stride", "before", "after"])
DenseNetConfiguration=namedtuple("DenseNetConfiguration",["inputChannel", "channel", "level", "kernel", "stride"])
AdamConfiguration=namedtuple("AdamConfiguration", ["rate", "beta"])

#Queue Block Configuration
queueBlockConfiguration=QueueBlockConfiguration(intermediateChannel=32, depthKernel=3, heightKernel=3, widthKernel=3, stride=1)

#Intermediate CNN Model Configuration
intermediateCNNConfiguration=IntermediateCNNConfiguration(inputChannel=1, channel=32, factor=4, kernel=3, stride=1, before=6, after=3)

