import numpy as np
from copy import deepcopy


def zeroR(target: np.array, new_data: np.array) -> int:
    '''
    Predict label based on zeroR algorithm
    :param target: the data use to learn
    :param new_data: the data to predict
    :return: the index of the label
    '''
    maxi = [0]*(np.max(target)+1)
    for ii,val in enumerate(target):
        maxi[val] += 1
    return np.argmax(maxi)

def AccuracyZeroR(data: np.array, target: np.array)-> float:
    '''
    compute the accuracy of the zeroR algorithm
    :param data: the data to use to learn
    :param target: the label of data
    :return: the accuracy of the zeroR algorithm
    '''
    accuracy = 0
    for indiceName in target :
        if zeroR(target,np.array([0])) == indiceName:
            accuracy += 1
    return accuracy/len(target)

def nbUniqueTarget(target: np.array) -> int:
    '''
    count the number of unique values in target array
    :param target: the target array
    :return: the number of unique values
    '''
    _, counts = np.unique(target, return_counts=True)
    return len(counts)


def discretizeValue(data: np.array, target: np.array) -> np.array:
    '''
    discretize the data in multiple interval based on number of unique values of target
    :param data: the data to discretize
    :param target: the label array
    :return: data discretized
    '''
    allDiscretInterval = []
    for ii in range(data.shape[1]):
        # Interval discretize
        discretisation = np.array([])

        feature_i = data[:, ii]

        # get the number of unique value
        uniqueTarget = nbUniqueTarget(target)

        pas = feature_i.mean() / uniqueTarget
        maxi = np.max(feature_i)
        mini = np.min(feature_i)
        discretisation = []
        while (mini + pas <= maxi):
            discretisation.append([mini, mini + pas])
            mini += pas  # new min not the same as the ancient max
        discretisation.append(np.array([mini, maxi]))
        allDiscretInterval.append(discretisation)

    return allDiscretInterval

def ProbaOneR(count: np.array, nbTarget: int) -> float:
    '''
    return the probality of count using oneR algorithm
    :param count: the array to find the probability
    :param nbTarget: the number of unique labels
    :return:
    '''
    #Avoid divide by zero
    CONST = 0.0001
    return np.divide(count,(count.sum(axis=1)+CONST)[:,np.newaxis])





def predictFeatureOneR(data: np.array, target: np.array):
    '''
    predict the features of data using oneR algorithm
    :param data: data to use for learn
    :param target: label of data
    :return: the index of the label predict, the array of discrete interval,array of label  equivalent
    '''
    discretInterval = discretizeValue(data, target)

    countLabel = deepcopy(list(discretInterval))

    # Replace all value by zero
    UniqueTarget = nbUniqueTarget(target)
    for ii, c in enumerate(countLabel):  # for each label interval list
        for i in range(len(c)):  # for each interval
            c[i] = [0] * UniqueTarget
        countLabel[ii] = c
    for ii, features in enumerate(data):
        for jj, valueFeature in enumerate(features):
            for indiceInterval, interval in enumerate(discretInterval[jj]):
                bornInf = interval[0]
                bornSup = interval[1]

                if bornInf < valueFeature < bornSup:
                    x = 5
                    countLabel[jj][indiceInterval][target[ii]] += 1

    # calculation of probality of each feature
    selectFeature = -1
    maxiProb = -np.inf
    for iiSelectFeature, f in enumerate(countLabel):
        nbValue = ProbaOneR(np.array(f), UniqueTarget)

        prob = np.divide(nbValue.max(axis=1).sum(), len(nbValue))
        if prob > maxiProb:
            maxiProb = prob
            selectFeature = iiSelectFeature
    # set the feature to predict

    nbValue = ProbaOneR(np.array(countLabel[selectFeature]), UniqueTarget)

    featurePredictList = np.argmax(nbValue, axis=1)
    return selectFeature, discretInterval[selectFeature], featurePredictList

def oneR(selectFeature: np.array, discretInterval: np.array, featurePredictList: np.array, new_data: np.array):
    '''
    return the predict label for one data using oneR algorithm
    :param selectFeature: the label to use for prediction
    :param discretInterval: array interval of the label
    :param featurePredictList: array label equivalent interval
    :param new_data: the data to predict the label
    :return: the index of the label
    '''
    for ii,interval in enumerate(discretInterval):
        if interval[0] < new_data[selectFeature] and new_data[selectFeature] < interval[1]:
            return featurePredictList[ii]

def AccuracyOneR(dataTrain: np.array, targetTrain: np.array, dataDev: np.array, targetDev: np.array)-> float:
    '''
    compute the accuracy of the oneR algorithm
    :param dataTrain: data to use to learn
    :param targetTrain: label of dataTrain
    :param dataDev: data to use to predict
    :param targetDev: label of dataDev
    :return: the accuracy of the oneR algorithm
    '''
    selectFeature, discretInterval, featurePredictList = predictFeatureOneR(dataTrain, targetTrain)
    accuracy = 0
    for ii,indiceName in enumerate(targetDev) :
        prediction = oneR(selectFeature,discretInterval,featurePredictList, dataDev[ii])
        if prediction == indiceName:
            accuracy += 1
    return accuracy/len(targetDev)