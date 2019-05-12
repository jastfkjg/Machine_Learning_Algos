"""
CART(Classification and Resression Tree) is a special algo which can handle both classification and regression problem.
Here, we use CART for regression task
"""

import numpy as np

def loadDataSet(filename):
    data = []
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            line = map(float, line)
            data.append(line)
        return data

def binSplitData(data, feature, value):
    d1 = data[np.nonzero(data[:, feature] > value)[0], :][0]
    d2 = data[np.nonzero(data[:, feature] <= value)[0], :][0]
    return d1, d2

def createTree(data):
    pass

def regLeaf(data):
    return np.mean(data[:, -1])

def regErr(data):
    return np.var(data[:, -1]) * np.shape(dataSet)[0]

def chooseBestSplit(data, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(data[:, -1].T.tolist()[0])) == 1:
        # all value are equal
        return None, leafType(data)
    m, n = np.shape(data)
    S = errType(data)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(data[:, featIndex]):
            d1, d2 = binSplitData(data, featIndex, splitVal)
            if np.shape(d1)[0] < tolN or np.shape(d2)[0] < tolN:
                continue
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if S - bestS < tolS:
        return None, leafType(data)
    d1, d2 = binSplitData(data, bestIndex, bestValue)
    if (shape(d0)[0] < tolN) or (np.shape(d1)[0] < tolN):
        return None, leafType(data)
    return bestIndex, bestValue


