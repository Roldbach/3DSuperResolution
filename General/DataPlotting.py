import matplotlib.pyplot as plt

from General.Configuration import experimentConfiguration

def plotLoss(trainLoss, validationLoss, name, path=experimentConfiguration.projectPath+"/Plot"):
    epoch=[i+1 for i in range(len(trainLoss))]
    figure=plt.figure(dpi=300)
    plt.plot(epoch, trainLoss, label="Training Loss")
    plt.plot(epoch, validationLoss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss/Validation Loss")
    plt.title(name+" Loss")
    plt.legend()
    plt.show()
    figure.savefig(path+"/"+name+".png")

def plotImage(image, name, path=experimentConfiguration.projectPath+"/Plot"):
    figure=plt.figure(dpi=300)
    plt.grid(False)
    plt.axis("off")
    plt.imshow(image,cmap=plt.cm.gray)
    figure.savefig(path+"/"+name+".png")

#For local usage
def plot3DImage(image, name, path):
    figure=plt.figure(figsize=(10, 10))
    axis=plt.axes(projection='3d')
    axis.voxels(image, edgecolors='grey')
    plt.axis('off')
    plt.show()
    figure.savefig(path+"/"+name+".png")