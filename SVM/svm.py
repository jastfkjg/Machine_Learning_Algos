import random
import numpy as np

def loadDataSet(filename):
    data = []
    label = []
    with open(filename) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            data.append([float(lineArr[0]), float(lineArr[1])])
            label.append(float(lineArr[2]))
    return data, label

def randSelect(i, m):
    x = i
    while x == i:
        x = int(random.uniform(0, m))
    return x

def clipAlpha(a, high, low):
    if a > high:
        a = high
    if a < low:
        a = low
    return a

def smoSimple(data, classLabels, C, toler, maxIter):
    dataMat = np.mat(data)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMat)
    alphas = np.mat(zeros(m, 1))
    it = 0
    while it < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(np.multiply(alphas, labelMat).T * (dataMat * dataMat[i, :].T)) + b
            ei = fxi - float(labelMat[i])
            if (labelMat[i] * ei < -toler and alphas[i] < C) or (labelMat[i] * ei > toler \
                    and alphas[i] > 0):
                # random choise the second alpha
                j = randSelect(i, m)
                fxj = float(np.multiply(alphas, labelMat).T * (dataMat * dataMat[j, :].T)) + b
                ej = fxj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                eta = 2.0 * dataMat[i, :] * dataMatp[j, :].T - dataMat[i, :] * dataMat[i, :].T \
                        - dataMat[j, :] * dataMat[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= labelMat[j] * (ei - ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print(" j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - ei - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i, :] * \
                        dataMat[i,:].T - labelMat[j] * (alphas[j] - alphaJold) * \
                        dataMat[i, :] * dataMat[j, :].T
                b2 = b - ej - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i, :] * \
                        dataMat[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * \
                        dataMat[j, :] * dataMat[j, :].T
                if 0 < alphas[i] and C > alphas[i]:
                    b = b1
                elif a < alphas[j] and C > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter : %d i: %d, pairs changes %d" %(it, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            it += 1
        else:
            it = 0
    return b, alphas


def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = math.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('the kernel is not recognized')
    return K


