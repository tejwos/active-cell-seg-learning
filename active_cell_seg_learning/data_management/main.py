import data_management.folderManagement
import data_management.moveFiles

def create_AL_infrastructure(dict_args):    
    
    ## Copy Data from "data folder" to "MDB folder"
    data_management.moveFiles.copyFiles(src=dict_args['dataset_path'], trg=dict_args['MDB'])

    # Move Files to RDB
    # -> for UC calculation
    data_management.moveFiles.movePartRandom_MandI(src=dict_args['MDB'], trg=dict_args['RDB'], 
                                                random_number= dict_args['fin_data_size'] - dict_args['al_start_size'], seed=dict_args['general_seed'],
                                                    )
    #Move Files to WDB
    # -> for training
    data_management.moveFiles.movePartRandom_MandI(src=dict_args['MDB'], trg=dict_args['WDB'], 
                                                random_number= dict_args['al_start_size'], seed=dict_args['general_seed'],
                                                    )
    # Move Files to Test and Train DB (TDB, VDB)
    # -> for test and validation
    data_management.moveFiles.movePartRandom_MandI(src=dict_args['MDB'], trg=dict_args['TDB'], 
                                                random_number= int(dict_args['fin_data_size'] * dict_args['test_percent']) , seed=dict_args['general_seed'],
                                                    )
    data_management.moveFiles.movePartRandom_MandI(src=dict_args['MDB'], trg=dict_args['VDB'], 
                                                random_number= int(dict_args['fin_data_size'] * dict_args['test_percent']) , seed=dict_args['general_seed'],
                                                    )