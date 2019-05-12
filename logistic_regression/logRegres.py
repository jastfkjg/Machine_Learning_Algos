import numpy as np
import random
import math

def loadDataSet():
    data = []
    label = []
    with open('testSet.txt') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            data.append([1.0, float(lineArr[0]), float(lineArr[1])])
            label.append(int(lineArr[2]))
    return data, label

def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))

def gradAscent(data, classLabels):
    dataMat = np.mat(data)
    labelMat = np.mat(classLabels).transpose()

    m, n = np.shape(dataMat)

    alpha = 0.001
    steps = 500
    weights = np.ones((n, 1))
    for k in range(steps):
        h = sigmoid(dataMat * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMat.transpose() * error
    return weights

def stocGradAscent(dataMat, classLabels):
    # dataMat = np.mat(data)
    m, n = np.shape(dataMat)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMat[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMat[i]
    return weights

def stocBatchGradAscent(dataMat, classLabels, numIter=100):
    dataMat = np.array(dataMat)
    m, n = np.shape(dataMat)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0+j+i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMat[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]
            # del(dataIndex[randIndex])
    return weights

if __name__ == "__main__":
    data, label = loadDataSet()
    weights = stocBatchGradAscent(data, label)
    print("weights for logistic regression: ", weights)


