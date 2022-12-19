### Import
# general
from argparse import ArgumentParser
import os
import torch
import numpy as np
import torch.optim as optim
import pandas as pd

# this project
import data_management.main
import data_management.moveFiles
import data_management.folderManagement
import uc_utils.uncertaintyFunctions
import data_loading.dataset_loader
import model_uc.model_uc
import model_uc.train
import loss.loss
import uc_utils.main

# mlf
import mlflow
from mlf_core.mlf_core import MLFCore


### Get User Input

if __name__ == "__main__":

    ##################################
    ## Get / Set All Parameter 
    parser = ArgumentParser(description='Active Learning Simulation for the TissueNet with 2 Channels and 3 Classes; and Unet as Model.')
    # Get MLF Core Parameters
    parser.add_argument('--general_seed',  type=int,        default=0,          help='general random seed')
    parser.add_argument('--pytorch_seed',  type=int,        default=0,          help='random seed of all pytorch functions')
    parser.add_argument('--log_interval',  type=int,        default=100,        help='log interval of stdout')
    parser.add_argument('--save_interval', type=int,        default=3,          help='overead intervall for csv savepoints')
    # Get ML Parameters (for the Unet)
    parser.add_argument('--num_workers', type=int,          default=0,          help='number of workers (default: 8)')
    parser.add_argument('--lr', type=float,                 default=0.0001,     help='learning rate (default: 0.0001)')
    parser.add_argument('--test_percent', type=float,       default=0.10,       help='dataset percent for testing (default: 0.10)')
    parser.add_argument('--epochs', type=int,               default=10,         help='epochs before testing (default: 10)')
    parser.add_argument('--dataset_path', type=str,         default='data',     help='path to dataset (default: data)')
    parser.add_argument('--batch_size', type=int,           default=2,          help='size of batch (default: 2')
    parser.add_argument('--n_channels', type=int,           default=2,          help='number of input channels (default: 2)')
    parser.add_argument('--n_class', type=int,              default=3,          help='number of classes (default: 3)')
    parser.add_argument('--dropout_rate', type=float,       default=0.2,     help='dropout rate (default: 0.0002)')
    parser.add_argument('--mask_suffix', type=str,          default='_mask',    help='Suffix')
    parser.add_argument('--device', type=str,               default="cuda",     help='set cuda or cpu (default: cuda)')
    # Get AL Parameters
    parser.add_argument('--t_value', type=int,              default=10,         help='deep of monte carlo sampling  (default: 36)')
    parser.add_argument('--al_start_size', type=int,        default=10,         help='size for first training (default: 10)')
    parser.add_argument('--al_push_size', type=int,         default=10,         help='size of selection basec on UCV value (default: 10)')
    parser.add_argument('--fin_data_size', type = int,      default=200,        help='maximum number of samples for AL training (default: 200)')
    parser.add_argument('--simulation_path', type=str,      default='sim',      help='path to simulation (default: sim)')
    parser.add_argument('--img_rows', type=int,             default=256,        help='size of image, num of rows (default 256)')
    parser.add_argument('--img_cols', type=int,             default=256,        help='size of image, num of columns (default 256)')
    parser.add_argument('--random_mode', type=bool,         default=False,      help='If True: set UC Estimation to Random, for baseline estimation')

    # args=[] only for debug in Notebooks, Without it: leads to some  strange errors   -> SystemExit: 2
    args = parser.parse_args(args=[])
    dict_args = vars(args)
    dict_args['result_list'] = []
    dict_args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    dict_args['size_last_iter'] = uc_utils.uncertaintyFunctions.get_size_of_last_iteration(fin_data_size=dict_args['fin_data_size'], 
                                                                            al_push_size=dict_args['al_push_size'],
                                                                            al_start_size=dict_args['al_start_size'],
                                                                            )

    ##################################
    ## MLF 
    # Log Part
    #mlflow.autolog(1)
    #MLFCore.log_sys_intel_conda_env()

    ## Deter. part
    MLFCore.set_general_random_seeds(dict_args['general_seed'])
    MLFCore.set_pytorch_random_seeds(dict_args['pytorch_seed'], dict_args['num_workers'])

    ##################################
    ## Create Infrastructure folder 
    # Create folders
    main_path_AL, subFolderList_export = data_management.folderManagement.createFolders(dict_args['simulation_path'], 'main')
    # add Paths of Sub-Folders to main dict
    dict_args = dict_args | subFolderList_export
    # copy and move files based on user input
    data_management.main.create_AL_infrastructure(dict_args)

    ##################################
    ## Start Active Learning Loop
    dict_args['num_steps'] = uc_utils.uncertaintyFunctions.get_number_of_steps_al(dict_args)

    for i in range(int(dict_args['num_steps'])):
        print(' ---> Active Lerning Step: ' , i)
    
        ##################################
        ## Set and Get DataLoader, Model and co.
        train_data_loader, test_data_loader, validation_data_loader = data_loading.dataset_loader.get_loaders(dict_args=dict_args)
        model = model_uc.model_uc.UNet2D(n_channels=dict_args['n_channels'],
                                    n_classes=dict_args['n_class'], 
                                    dropout_val=dict_args['dropout_rate']).to(dict_args['device'])
        
        lossFn = loss.loss.FocalLoss()
        
        if torch.cuda.is_available() and dict_args['device'] == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = 0

        optimizer = optim.Adam(model.parameters(), lr=dict_args['lr'])


        ##################################
        ## Train, Validation and Test

        model_uc.train.train_model(dict_args=dict_args, model=model, 
                                    train_loader=train_data_loader, 
                                    validation_loader=validation_data_loader, 
                                    optimizer=optimizer,
                                    lossFn=lossFn,
                                    scaler=scaler,
                                    i = i,
                                    )

        ##################################
        ## UC Estimation from RDB
        df_uc = uc_utils.main.active_learning_selection(model=model, dict_args=dict_args)

        ##################################
        ## File Moving
        data_management.moveFiles.moveFilesOnUncertainty_MandI(dict_args=dict_args, df=df_uc)

        ##################################
        ## Save Results in CSV
        df_results = pd.DataFrame(data=dict_args['result_list'])
        result_path = os.path.join(dict_args['Results'], 'AL_Result.csv')
        df_results.to_csv(result_path)


