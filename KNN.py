import itertools
from typing import Callable

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

from Rule import *

def KNN_Algo(datas: np.array, target: np.array, k, distance: Callable, new_data: np.ndarray) -> int:
    '''

    :param datas: data to use to learn
    :param target: label of data
    :param k: the number of neighbors
    :param distance: function to compute the distance
    :param new_data: data to find it label
    :return: the index of the label
    '''
    dicDist = {}
    for ii, data in enumerate(datas):
        dicDist[ii] = distance(data, new_data)
    # sorted by value
    listDist = dict(sorted(dicDist.items(), key=lambda item: item[1]))
    # get k first
    splitKNear = dict(itertools.islice(listDist.items(), k))

    # get the major cluster
    _, counts = np.unique(target, return_counts=True)
    listTarget = [0] * len(counts)
    for key in splitKNear:
        listTarget[target[key]] += 1
    return np.argmax(listTarget)


def accuracyKNN(X_train: np.array, y_train: np.array, data: np.array, target: np.array, k: int) -> float:
    '''
    compute the accuracy of the KNN algorithm
    :param data: data to use to learn
    :param target: label of data
    :param k: the number of neighbors
    :return: the accuracy of the KNN algorithm
    '''
    nbGoodAnswer = 0
    nbAnswer = data.shape[0]

    for ii in range(nbAnswer):
        res = KNN_Algo(X_train, y_train, k, distance, data[ii])
        if res == target[ii]:
            nbGoodAnswer += 1
    return nbGoodAnswer / nbAnswer

def distance(x1: np.ndarray, x2: np.ndarray) -> float:
    '''
    compute distance between two array
    :param x1: first array to compare
    :param x2: second array to compare
    :return: the distance between x1 and x2
    '''
    return np.linalg.norm(x1 - x2)

def toIrisName(iris: Bunch, nb: int)-> str:
    '''
    convert number [0 ; 2] to iris label name equivalent
    :param nb: feature's index
    :return: the equivalent name for iris data
    '''
    return iris["target_names"][nb]


def __main__ ():
    iris = load_iris()

    data = np.array(iris['data'])
    target = np.array(iris['target'])

    data = data[:, :]
    X_train, X_dev, y_train, y_dev = train_test_split(data, target, test_size=0.25, train_size=0.75, random_state=42)


    print(f"Accuracy of KNN with k=3 ,{accuracyKNN(X_train,y_train,X_dev,3)}")
    print(f"Accuracy of ZeroR ,{AccuracyZeroR(X_train, y_dev)}")
    print(f"Accuracy of OneR,{AccuracyOneR(data,target,X_dev,y_dev)}")

__main__()

