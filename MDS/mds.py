import numpy as np 
import os
import matplotlib.pyplot as plt


# 处理数据，获得原始数据
def processData(fileName):
    dataSet = []

    with open(fileName, "r") as f:
        for line in f:
            if len(line) < 10: continue 
            line = line.rstrip("\n").split(",")
            data = list(map(float, line[:-1]))
            if line[-1] == "Iris-setosa":
                data.append(0)
            elif line[-1] == "Iris-versicolor":
                data.append(1)
            else:
                data.append(2)
            dataSet.append(data)

    return np.array(dataSet)

#  处理原始数据，获得高维空间下的实对称距离矩阵 D
def dist(dataSet):
    nums = len(dataSet)
    D = [[float("inf") for _ in range(nums)] for _ in range(nums)]

    for i in range(nums):
        for j in range(nums):
            D[i][j] = L2(dataSet[i], dataSet[j])

    return np.array(D)

# L^2范数， 欧几里得距离
def L2(x, y):
    x, y = x[:-1], y[:-1]
    # 这里最后一位是标签，注意不要算进距离
    distance = np.sqrt(np.sum(np.square(x - y)))
    return distance

# 内积矩阵B
def calculateB(D):
    nums = D.shape[0]
    B = [[float("inf") for _ in range(nums)] for _ in range(nums)]

    dist = np.mean(np.mean(np.square(D))) 
    for i in range(nums):
        for j in range(nums):
            distij = np.square(D[i][j])
            disti = np.mean(np.square(D[i,:]))
            distj = np.mean(np.square(D[:,j]))
            B[i][j] = -1.0 / 2 * (distij - disti - distj + dist) 
    
    return np.array(B)

# 特征值分解
def EIG(matrix):
    eVals, eVecs = np.linalg.eig(matrix)
    return eVals, eVecs

# 构建数据的低纬表示
def reStructure(eVals, eVecs):
    # iris数据原本是4维的，这里现将数据转为2维
    return np.dot(eVecs[:,:2], np.diag(np.sqrt(eVals[:2])))

# 比较高维数据和低纬数据之间的距离差异
def differences(dataSet, lowDimDataSet):
    nums = len(dataSet)

    for i in range(nums):
        for j in range(i+1, nums):
            print(np.sqrt(np.sum(np.square(dataSet[i] - dataSet[j]))), end = "    ")
            print(np.sqrt(np.sum(np.square(lowDimDataSet[i] - lowDimDataSet[j]))))

def draw(lowDimDataSet):
    setosa = lowDimDataSet[0:50]
    versicolor = lowDimDataSet[50:100]
    virginica = lowDimDataSet[100:]

    plt.scatter(setosa[:, 0], setosa[:, 1], c = "r")
    plt.scatter(versicolor[:, 0], versicolor[:, 1] ,c = "g")
    plt.scatter(virginica[:, 0], virginica[:, 1], c = "b")
    plt.show()

if __name__ == "__main__":
    fileName = "iris.data"
    # 处理数据
    dataSet = processData(fileName)
    # 获取实对称距离矩阵D
    D = dist(dataSet)
    # 获取内积矩阵B
    B = calculateB(D)
    # 获取特征值分解后的特征值和特征向量
    eVals, eVecs = EIG(B)
    # 获取低纬表示的数据集
    lowDimDataSet = reStructure(eVals, eVecs)
    # 测试降维前后的距离差异
    # differences(dataSet, lowDimDataSet)
    # 将数据画出来
    draw(lowDimDataSet)
