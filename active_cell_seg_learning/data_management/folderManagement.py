import os

def createFolders(path, folderName):
    """
    The subfolder Infrastructure for Active Learning will be created
    MDB: Main DataBase
    WDB: Working Database
    RDB: Resting (waiting for )
    """

    main_path = path
    main_path = os.path.join(main_path, folderName)
    
    ### check if main folders exist:
    if os.path.exists(main_path):
        pass
    else:
        os.makedirs(main_path)

    ### check if subfolders exist, if not, create them
    if False: #os.path.exists(os.path.join(main_path, "MDB")):
        pass
    else:
        subFolderList = ["MDB", "WDB", "RDB", "TDB", "VDB" ]
        subFolderImages = ["Imgs", "Msks"]
        subFolderList_export = {}
        for element in subFolderList:
            os.makedirs(os.path.join(main_path, element))
            temp = os.path.join(main_path, element)
            subFolderList_export[element] = temp
            for subelement in subFolderImages:
                temp = os.path.join(main_path, element, subelement)
                os.makedirs(temp)

        temp = os.path.join(main_path,"Results")
        os.makedirs(temp)
        subFolderList_export['Results'] = temp

    return main_path, subFolderList_export
   

def createFolder(path):
    """
    Create Folder
    """
    main_path = path

    ### check if main folders exist:
    if os.path.exists(main_path):
        pass
    else:
        os.makedirs(main_path)

def createSubfolder(path, subfolderName):
    """
    Create Subfolder
    """
    main_path = path
    main_path = os.path.join(main_path, subfolderName)

    ### check if main folders exist:
    if os.path.exists(main_path):
        pass
    else:
        os.makedirs(main_path)