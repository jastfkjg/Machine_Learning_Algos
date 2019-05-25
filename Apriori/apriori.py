
import numpy as np

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createOne(dataset):
    """
    生成1个项组成的候选项集的列表
    """
    oneElement = []
    for d in dataset:
        for item in d:
            if not [item] in oneElement:
                oneElement.append([item])
    oneElement.sort()
    return map(frozenset, oneElement)

def scan(dataset, elementList, minSupport):
    countDict = {}
    for d in dataset:
        for ele in elementList:
            if ele.issubset(d):
                if ele not in countDict:
                    countDict[ele] = 1
                else:
                    countDict[ele] += 1

    num = float(len(dataset))
    filterList = []
    supportData = {}
    for key in countDict:
        support = countDict[key] / num
        if support >= minSupport:
            filterList.append(key)
        supportData[key] = support

    return filterList, supportData

def aprioriGen(kElementList, k):
    retList = []
    lenth = len(kElementList)
    for i in range(length):
        for j in range(i+1, length):
            l1 = list(kElementList[i][:k-2])
            l2 = list(kElementList[j][:k-2])
            l1.sort()
            l2.sort()
            if l1 == l2:
                retList.append(kElementList[i] | kElementList[j])
    return retList

def apriori(dataset, minSupport=0.5):
    oneElementList = createOne(dataset)
    d = map(set, dataset)
    onefilterList, supportData = scan(d, oneElementList, minSupport)
    kElementList = [onefilterList]
    k = 2
    while kElementList[k-2]:
        kElementList = aprioriGen(kElementList[k-2], k)
        kfilterList, ksupportData = scan(d, kElementList, minSupport)
        supportData.update(ksupportData)
        kElementList.append(kfilterList)
        k += 1
    return kElementList, supportData






