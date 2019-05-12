
import numpy as np 

def loadDataSet(fileName, delim='\t'):
    with open(fileName) as fr:
        stringArr = [line.strip().split(delim) for line in fr.readlines()]
        datArr = [map(float,line) for line in stringArr]
    return np.mat(datArr)


