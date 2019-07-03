import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA

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

def myPCA(dataSet):
    matrix = np.dot(dataSet.T, dataSet)
    eVals, eVecs = np.linalg.eig(matrix)

    return np.dot(dataSet, eVecs[:,:2])

def draw(dataSet1, dataSet2):
    plt.figure()

    plt.subplot(121)
    plt.scatter(dataSet1[:50,0], dataSet1[:50,1], c = "r")
    plt.scatter(dataSet1[50:100,0], dataSet1[50:100,1], c = "g")
    plt.scatter(dataSet1[100:150,0], dataSet1[100:150,1], c = "b")

    plt.subplot(122)
    plt.scatter(dataSet2[:50,0], dataSet2[:50,1], c = "r")
    plt.scatter(dataSet2[50:100,0], dataSet2[50:100,1], c = "g")
    plt.scatter(dataSet2[100:150,0], dataSet2[100:150,1], c = "b")

    plt.show()

if __name__ == "__main__":
    fileName = "iris.data"
    # 获取原始数据
    dataSet = dataProcess(fileName)
    # 获取原始数据的低纬表示
    lowDimDataSet = myPCA(dataSet)
    # sklearn中自带PCA方法
    pca = PCA(2, True)
    testData = pca.fit_transform(dataSet)
    # 将数据画出来
    draw(lowDimDataSet, testData)
