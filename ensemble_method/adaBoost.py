import numpy as np

def loadData():
    data = np.matrix([[1., 2.1], [2., 1.1], [1.2, 1.], [1., 1.], [2., 1.]])
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return data, labels

def classify(data, dim, threshold):
    ret = np.ones((np.shape(data)[0], 1))
    ret[data[:, dim] <= threshold] = -1.0
    return ret 

def buildDT(data, labels, D):
    data = np.mat(data)
    labels = np.mat(labels).T
    steps = 10.0
    best = {}
    minError = np.inf
    for i in range(n):
        rangeMin = data[:, i].min()
        rangeMax = data[:. i].max()
        stepSize = (rangeMax - rangeMin) / steps

        for j in range(-1, int(steps) + 1):
            pass

def adaBoostTrain(data, labels, numIter=40):
    weakClass = []
    m = np.shape(data)[0]
    D = np.mat(ones((m, 1)) / m)
    

