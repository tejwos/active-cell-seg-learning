import os
import numpy as np

#def getUCValuesList(listOfElements, path, model):
#    """
#    Get Uncertainty Values for each Element in a given List.
#    """#
##
 #   resultList = []
#
 #   srcPath = path
  #  subFolderImages = ["Imgs", "Msks"]
   ## for element in listOfElements:
   #     imgFile = os.path.join(srcPath, subFolderImages[0], element)#
#
 #       resultList.append(getUCValue(imgFile, model))
#
  #  return resultList


def getUCValue(imgFullPath, model):
    """
    Creates an Monte Carlo Prediction Stack for an Image with deep T and get UC Value for this.
    """

    shortCut = np.random.random()
    return shortCut

def uncertaintyEstimation_PSD(predictionStack, img_rows, img_cols, numClasses=2):
    '''
    For a given Stack of Monte Carlo Predictions, an UC Value and a UC Map will be created.
    '''

    ### Count each Class for each Pixel Value
    ucPreMatrix = np.zeros([1, img_rows, img_cols])
    for i in range(numClasses):
        count = np.count_nonzero(predictionStack == i, axis=0)
        count = np.expand_dims(count, axis=0)
        #count = np.sum(predictionStack == i, axis=0)
        ucPreMatrix = np.concatenate((ucPreMatrix, count))
    ucPreMatrix = np.delete(ucPreMatrix, [0], 0)

    ### Calculate Sum of all Values for each Pixel
    ucSum = np.sum(ucPreMatrix, axis = 0)

    ### Get Probability of each Class per Pixel Stack
    ucPreMatrix_relativ = np.divide(ucPreMatrix, ucSum)

    ### Get Std of Probabilities of each Class per Pixel Stack
    ucPreMatrix_std = np.std(ucPreMatrix_relativ, axis=0)

    ### Estimation of Max Std for given Number of Classes
    maxStdEstimation = np.zeros([numClasses])
    maxStdEstimation[0] = 1
    maxStdEstimation = np.std(maxStdEstimation)

    ### Normalization each Pixel based on Max Std (= UC Map)
    ucMap = np.divide( np.subtract(maxStdEstimation , ucPreMatrix_std ), maxStdEstimation) 
    ucValue = np.mean(ucMap)
    return ucValue, ucMap

def uncertaintyEstimation_PSD_simulation(numClasses=2, img_rows=256, img_cols=256, T = 100):
    predictionStack = np.zeros([1, img_rows, img_cols])
    predictionStack = np.random.randint(0,high=numClasses, size=[T, img_rows, img_cols])

    ### Count each Class for each Pixel Value
    ucPreMatrix = np.zeros([1, img_rows, img_cols])
    for i in range(numClasses):
        count = np.count_nonzero(predictionStack == i, axis=0)
        count = np.expand_dims(count, axis=0)
        #count = np.sum(predictionStack == i, axis=0)
        ucPreMatrix = np.concatenate((ucPreMatrix, count))
    ucPreMatrix = np.delete(ucPreMatrix, [0], 0)

    ### Calculate Sum of all Values for each Pixel
    ucSum = np.sum(ucPreMatrix, axis = 0)

    ### Get Probability of each Class per Pixel Stack
    ucPreMatrix_relativ = np.divide(ucPreMatrix, ucSum)

    ### Get Std of Probabilities of each Class per Pixel Stack
    ucPreMatrix_std = np.std(ucPreMatrix_relativ, axis=0)

    ### Estimation of Max Std for given Number of Classes
    maxStdEstimation = np.zeros([numClasses])
    maxStdEstimation[0] = 1
    maxStdEstimation = np.std(maxStdEstimation)

    ### Normalization each Pixel based on Max Std (= UC Map)
    ucMap = np.divide( np.subtract(maxStdEstimation , ucPreMatrix_std ), maxStdEstimation) 
    ucValue = np.mean(ucMap)
    return ucValue, ucMap

def rangeTransform(sample):
    """
    Normalization for 255 range of values
    :param sample: numpy array for normalize
    :return: normalize array
    """
    if (np.max(sample) == 1):
        sample = sample * 255

    m = 255 / (np.max(sample) - np.min(sample))
    n = 255 - m * np.max(sample)
    return (m * sample + n) / 255

def enable_dropout(model):
    '''
    Dropout Layers to train() with rest of model is in eval()
    '''
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
          m.train()


def get_size_of_last_iteration(fin_data_size = 200, al_push_size=25, al_start_size=13 ):
    '''
    :return: size of last push-set, if push_size % (findata - start_size) != 0 
    '''
    a = fin_data_size
    b = al_start_size
    c = al_push_size
    d = a - b
    e = d % c
    f = int(d / c)
    #print(e + b + c*f )
    return e

def get_number_of_steps_al(dict_args):
    num_steps = np.floor( np.divide(np.subtract(dict_args['fin_data_size'], dict_args['al_start_size']), dict_args['al_push_size']))
    return num_steps