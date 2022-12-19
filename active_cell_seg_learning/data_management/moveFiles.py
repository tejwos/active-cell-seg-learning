import shutil
import os

import numpy as np
import pandas as pd

import uncertaintyFunctions

def copyFiles(src:str, trg:str):
    """
    Creats a Copy of all Files in a given Folder with Subfolders "Imgs" & "Msks".
    Only works for this 2 Subfolders!
    """
    srcPath = src
    destPath = trg
    subFolderImages = ["Imgs", "Msks"]

    if os.path.exists(srcPath) and os.path.exists(destPath):
        pass
    else:
        raise ValueError("A path in copyFiles() is not valide. ")
        
    for element in subFolderImages:
        tempSrcPath = os.path.join(srcPath, element)
        files = os.listdir(tempSrcPath)
        tempDestPath = os.path.join(destPath, element)
        for file_name in files:
            full_file_name = os.path.join(tempSrcPath, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, tempDestPath)


def copyPartRandom_MandI(src:str , trg:str, number:int, seed:int):
    """
    Move random Subset of Files in a given Folder with Subfolders "Imgs" & "Msks".
    Only works for this 2 Subfolders!
    """
    srcPath = src
    destPath = trg
    subFolderImages = ["Imgs", "Msks"]

    if os.path.exists(srcPath) and os.path.exists(destPath):
        pass
    else:
        raise ValueError("A path in copyPartRandom_MandI() is not valide. ")

    ## Create Lists 
    filesImg = sorted(os.listdir(os.path.join(srcPath, subFolderImages[0])), key = lambda element: element.split("_")[1] )
    filesMsk = sorted(os.listdir(os.path.join(srcPath, subFolderImages[1])), key = lambda element: element.split("_")[1] )
    if len(filesImg) == 0:
        print("No File left for movePartRandom_MandI().")
        return
        
    ## Use list for random selection
    list = [filesImg,filesMsk]
    npArray = np.array(list)
    df = pd.DataFrame(npArray)

    df2 = df.T.sample(number, random_state=seed, replace=False)
    #print(df2)
    i = 0
    for element in subFolderImages:
        tempSrcPath = os.path.join(srcPath, element)
        tempDestPath = os.path.join(destPath, element)

        for file_name in df2.iloc[:,i]:
            #print(file_name)
            full_file_name = os.path.join(tempSrcPath, file_name)
            if os.path.isfile(full_file_name):
                full_file_trg = os.path.join(tempDestPath, file_name)
                shutil.copy(full_file_name, full_file_trg)
        i = i + 1


def movePartRandom_MandI(src, trg, random_number, seed):
    """
    Move random Subset of Files in a given Folder with Subfolders "Imgs" & "Msks".
    Only works for this 2 Subfolders!
    """

    ## example
    #srcPath = r"C:\Users\Ing_W\Desktop\Master Thesis\data\toyDB_subset"
    srcPath = src

    # path to destination directory
    destPath = trg
    
    subFolderImages = ["Imgs", "Msks"]

    if os.path.exists(srcPath) and os.path.exists(destPath):
        pass
    else:
        raise ValueError("A path in movePartRandom_MandI() is not valide. ")

    ## Create Lists 
    
    filesImg = sorted(os.listdir(os.path.join(srcPath, subFolderImages[0])), key = lambda element: element.split("_")[1] )
    filesMsk = sorted(os.listdir(os.path.join(srcPath, subFolderImages[1])), key = lambda element: element.split("_")[1] )
    #print(filesMsk)
    #print(filesImg)
    if len(filesImg) == 0:
        print("No File left for movePartRandom_MandI().")
        return
        

    ## Use list for random selection
    list = [filesImg,filesMsk]
    #print(len(filesImg))
    #print(len(filesMsk))
    npArray = np.array(list)
    df = pd.DataFrame(npArray)


    df2 = df.T.sample(random_number, random_state=seed, replace=False)

    i = 0
    for element in subFolderImages:
        tempSrcPath = os.path.join(srcPath, element)
        tempDestPath = os.path.join(destPath, element)

        for file_name in df2.iloc[:,i]:
            #print(file_name)
            full_file_name = os.path.join(tempSrcPath, file_name)
            if os.path.isfile(full_file_name):
                full_file_trg = os.path.join(tempDestPath, file_name)
                shutil.move(full_file_name, full_file_trg)
            
        i = i + 1


def moveAllFiles(src, dest):
    """
    Move all Files in a given Folder with Subfolders "Imgs" & "Msks".
    Only works for this 2 Subfolders!
    """

    ## example
    #srcPath = r"C:\Users\Ing_W\Desktop\Master Thesis\data\toyDB_subset"
    srcPath = src

    # path to destination directory
    destPath = dest
    
    subFolderImages = ["Imgs", "Msks"]

    if os.path.exists(srcPath) and os.path.exists(destPath):
        pass
    else:
        raise ValueError("A path in copyFiles() is not valide. ")
        
    for element in subFolderImages:
        tempSrcPath = os.path.join(srcPath, element)
        files = os.listdir(tempSrcPath)
        #print(files)
        tempDestPath = os.path.join(destPath, element)

        for file_name in files:
            full_file_name = os.path.join(tempSrcPath, file_name)
            if os.path.isfile(full_file_name):
                shutil.move(full_file_name, tempDestPath)

def moveFilesOnUncertainty_MandI(dict_args, df):
    """
    Move subset of Files in a given Folder with Subfolders "Imgs" & "Msks".
    - Only works for this 2 Subfolders! 
    """
    src_path = dict_args['RDB']
    maskPath = 'Msks'
    imgsPath = 'Imgs'

    trg_path = dict_args['WDB']

    for image, mask in zip(df.loc[:, 1], df.loc[:,2]):
        #print(image, mask)
        src_image_path = os.path.join(src_path, imgsPath, image)
        trg_image_path = os.path.join(trg_path, imgsPath, image)
        shutil.move(src_image_path, trg_image_path)

        src_mask_path = os.path.join(src_path, maskPath, mask)
        trg_mask_path = os.path.join(trg_path, maskPath, mask)
        shutil.move(src_mask_path, trg_mask_path)



        

    

################################################################
# working part ################
################################################################

def test():
    pass 
