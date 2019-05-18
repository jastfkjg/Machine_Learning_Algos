
import numpy as np 

def loadDataSet(fileName, delim='\t'):
    with open(fileName) as fr:
        stringArr = [line.strip().split(delim) for line in fr.readlines()]
        datArr = [map(float,line) for line in stringArr]
    return np.mat(datArr)

def pca(data, topNfeat):
    """
    topNfeat: how many eigen vectors to take, or dim of new space
    """
    mean = np.mean(data, axis=0)
    meanRemoved = data - mean
    # calculate covariance matrix
    cov = np.cov(meanRemoved, rowvar=0)
    # calculate eigen values and eigen vectors
    eigvals, eigVects = np.linalg.eig(np.mat(cov))
    # sort according to eigen values
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:, eigValInd]
    lowData = meanRemoved * redEigVects
    recon = (lowData * redEigVects) + mean
    return lowData, recon 




