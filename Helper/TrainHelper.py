import os
import shutil



def countEpoch(name, path):
    '''
        Return the current epoch number based on the
    loss data in the file
    '''
    try:
        with open(path+"/"+name+".txt", "r") as file:
            lines=file.readlines()
        return len(lines)
    except FileNotFoundError:
        print("The file was not successfully loaded. Please try again.")

def cleanResult(path):
    '''
        If not in loading mode but the directory
    already exists, this must be the unwanted result
    from the last experiment and should be cleaned
    '''
    if os.path.isdir(path):
        shutil.rmtree(path)
