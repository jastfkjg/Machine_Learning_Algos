import numpy as np
import operator


def createDataSet():
    dataSet = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.1], [0.1, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return dataSet, labels

def normalize(dataSet):
    # newValue = (oldValue - min) / (max - min) 
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet


def classify(X, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(X, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # argsort(): return the indices that sort an array
    sortedDistIndices = distances.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]



if __name__ == "__main__":
    dataSet, labels = createDataSet()
    print("our training dataset is: ", dataSet)
    print("with labels: ", labels)
    normDataSet = normalize(dataSet)
    # print(normDataSet)
    test = [0, 0]
    print("test point: ", test)
    label = classify(test, normDataSet, labels, k=3)
    print("result: ", label)
    

