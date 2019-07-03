import numpy as np 
import matplotlib.pyplot as plt 

def dataProcess(fileName):
    dataSet = []

    with open(fileName, "r") as f:
        for line in f:
            if len(line) < 10: continue
            line = line.rstrip("\n").split(",")
            dataSet.append(list(map(float, line[:-1])))

    dataSet = np.array(dataSet)
    # 数据去中心化
    dataSet -= np.mean(dataSet, axis = 0)
    return dataSet

def PCA(dataSet):
    # matrix = np.cov(dataSet.T, ddof = 0) 
    matrix = np.dot(dataSet.T, dataSet)
    eVals, eVecs = np.linalg.eig(matrix)
    kLargeIndex = eVals.argsort()[-2:][::-1]
    kLargeVecs = eVecs[kLargeIndex]

    return np.dot(dataSet, kLargeVecs.T)

def draw(dataSet):
    plt.scatter(dataSet[:50,0], dataSet[:50,1], c = "r")
    plt.scatter(dataSet[50:100,0], dataSet[50:100,1], c = "g")
    plt.scatter(dataSet[100:150,0], dataSet[100:150,1], c = "b")
    plt.show()

if __name__ == "__main__":
    fileName = "iris.data"
    # 获取原始数据
    dataSet = dataProcess(fileName)
    # 获取原始数据的低纬表示
    lowDimDataSet = PCA(dataSet)
    # 将低纬数据画出来 
    draw(lowDimDataSet)