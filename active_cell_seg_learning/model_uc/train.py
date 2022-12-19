import os
import pandas as pd

from tqdm import tqdm
import torch
import torch.nn as nn

import model_uc.validation
import model_uc.test


def train_model(dict_args, model, train_loader, validation_loader, optimizer, lossFn, scaler, i):
    '''
    Training and Validation per Epoch
    '''
    save_path = dict_args['Results']

    ### Create for Saving Results
    epochLossValues = []
    epochLossValuesValidation = []
    testValues = []
    dfPathTest = os.path.join(save_path,'_TrainingLoss_loopstep_' + str(i) + '.csv')
    dfPathValidation = os.path.join(save_path,'_ValidationLoss_loopstep_' + str(i) + '.csv')
    dfPathTest = os.path.join(save_path,'_TestLoss_loopstep_' + str(i) + '.csv')

    ### Train and Validation for each Epoch
    for epoch in range(dict_args['epochs']):
        
        ### Train Model
        model.train()
        epochLoss = train_model_losscalc( 
                        dict_args=dict_args,
                        loader=train_loader, 
                        model=model, 
                        optimizer=optimizer, 
                        lossFunction= lossFn, 
                        scaler=scaler
                        )
        
        
        ### Model Validation / Evaluation
        model.eval()

        ### Get Accuracy for Validation:
        epochLossValues.append(epochLoss / len(train_loader))
        validationValues = model_uc.validation.validation(loader=validation_loader, model=model, device=dict_args['device'], classes=dict_args['n_class'])
        epochLossValuesValidation.append(validationValues)
        df_Training = pd.DataFrame(data={'TrainingLoss': epochLossValues})
        df_Validation = pd.DataFrame(data=epochLossValuesValidation, columns=['Validation Lose Score', 'Accuracy', 'Right Pixels', "IoU Mean"])

        ### Save
        #if df_Training['TrainingLoss'].iloc[epoch] <= df_Training['TrainingLoss'].min():
        #    save_path(model, path=save_path)
        #    savePredictionsAsImgs(loader=validation_loader, model=model, path=save_path, device=DEVICE)

        ### Save Loss Values
        if epoch % 3 == 0 or epoch == dict_args['epochs']:
                df_Training = pd.DataFrame(data={'TrainingLoss': epochLossValues})
                df_Training.to_csv(dfPathTest) 
                df_Validation = pd.DataFrame(data=epochLossValuesValidation, 
                                    columns=['Validation Lose Score', 'Accuracy', 'Right Pixels','IoU' ])
                df_Validation.to_csv(dfPathValidation)

    ### Testing Model
    model.eval()
    testValues = model_uc.test.test(loader=validation_loader, model=model, device=dict_args['device'], classes=dict_args['n_class'])

    temp = dict_args['result_list']  
    temp.append(testValues)
    dict_args['result_list'] = temp

    testValues = [testValues]

    df_Test = pd.DataFrame(data=testValues, columns=['Validation Lose Score', 'Accuracy', 'Right Pixels', "IoU Mean"])
    df_Test.to_csv(dfPathTest)




def train_model_losscalc(dict_args, loader, model, optimizer, lossFunction, scaler=0):
    
    loop = tqdm(loader)
    runningLoss = 0

    ### Batch Loop in each Epoch
    for batch_idx, (data, targets) in enumerate(loop):
        data        = data.to(device=dict_args['device'])
        #targets     = targets.float().unsqueeze(1).to(device=DEVICE) #old!!! for 1 class
        targets     = targets.type(torch.LongTensor).to(device=dict_args['device'])

        ### Forward
        if dict_args['device'] == "cuda":
            with torch.cuda.amp.autocast():
                predictions = model(data) 
                loss = lossFunction(predictions, targets, classes=dict_args['n_class'])
        else:
            predictions = model(data)
            loss = lossFunction(predictions, targets, classes=dict_args['n_class'])


        ### backward
        if dict_args['device'] == "cuda" and scaler != 0:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:            
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

        #update tqdm loop
        runningLoss += loss.item()
        loop.set_postfix(loss=loss.item())
    return runningLoss