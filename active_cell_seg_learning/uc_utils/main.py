
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import numpy as np
import os
import pandas as pd

import uc_utils.uncertaintyFunctions
import data_loading.dataset_loader

def active_learning_selection(model, dict_args):
    model.eval()
    uc_utils.uncertaintyFunctions.enable_dropout(model)

    validationPath = dict_args['RDB']
    validationMaskSuffix = dict_args['mask_suffix']
    nChannels = dict_args['n_channels']
    nClass = dict_args['n_class']
    imgRows = dict_args['img_rows']
    imgCols = dict_args['img_cols']

    validationTransform = A.Compose(
        [
            A.Resize(height=imgRows, width=imgCols),
            A.Normalize(
                mean = [0] * nChannels, 
                std = [1] * nChannels, 
                max_pixel_value=255.0, 
            ),
            ToTensorV2()
        ],
    )

    uncertaintyDataLoader = data_loading.dataset_loader.get_loader(
        path=validationPath,
        mask_suffix=validationMaskSuffix,
        transform=validationTransform,
        batch_size = 1,
        number_worker = 1,
        pin_memory=False,
        shuffel=False,
    )

    #get list of Elements in RDB
    inputPath  = os.path.join(validationPath, "Imgs")
    targetPath = os.path.join(validationPath, "Msks")
    inputsList     = sorted(os.listdir(inputPath), key = lambda element: element.split("_")[1])
    targetsList    = sorted(os.listdir(targetPath), key = lambda element: element.split("_")[1])

    result_list = []

    for index, (x, y) in enumerate(uncertaintyDataLoader):
        #print(index)

        ### Create Stack of Predictions for Monte Carlo Approx. for one Input-Image
        # x:= Stack of Predictions 
        X = np.zeros([1, imgRows, imgCols])
        for t in range(dict_args['t_value'] ):    
            
            x = x.to(device=dict_args['device'])
            with torch.no_grad():
                logits = model(x) # z := logits prediction

            predictions = torch.softmax(logits, dim=1) ### NEW here: was: model(data) now torch.softmax(model(data))
            predictions = torch.argmax(predictions,  dim=1) 
            ## convert Torch Tensor to numpy array
            predictions = predictions.detach().numpy()
            # delete batch dimension
            #predictions = predictions[0]
            X = np.concatenate((X, predictions))
        # delete first zero 
        X = np.delete(X, [0], 0)
        #print(np.unique( X) )  # <------------------------ For Epoch Estimation it later
        ## Get UC Map and Values
        if dict_args['random_mode']  == False:  
            uc_value, uc_map = uc_utils.uncertaintyFunctions.uncertaintyEstimation_PSD(predictionStack=X, img_rows=imgRows, img_cols=imgCols, numClasses=nClass)
            result_list.append([index, inputsList[index], targetsList[index], uc_value, uc_map] )
        else:
            uc_value = np.random.randint(0, 100)
            result_list.append()

        #print(result_list)
    df = pd.DataFrame(data = result_list)
    df = df.sort_values(3, ascending=False).head(dict_args['al_push_size'])
    return df



    



