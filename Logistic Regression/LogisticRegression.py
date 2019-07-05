import numpy as np
import random, os
import matplotlib.pyplot as plt

def logisticFunction(z):
    return 1.0 / (1 + np.exp(-z))

def lossFunction(x, y, w):
    return -np.dot(y.T, np.dot(x, w.T)) + np.sum(np.log(1 + np.exp(np.dot(x, w.T))))

def logisticRegression(trainData, trainDataLabel):
    w = np.ones((1, np.shape(trainData)[1]))
    learningRate = 0.001
    times = 1000
    # 画图所需
    loss_sor = []
    time_sor = []
    # path = os.path.dirname(__file__)
    # os.chdir(path)

    for i in range(times):
        loss = lossFunction(trainData, trainDataLabel, w)
        loss_sor.extend(loss)
        time_sor.append(i+1)
        w -= learningRate * (np.dot(trainData.T, logisticFunction(np.dot(trainData, w.T)) - trainDataLabel)).T

    plt.xlabel("Times")
    plt.ylabel("Loss")
    plt.plot(time_sor, loss_sor)
    plt.show()
    return w

def logisticRegressionTest(testData, testDataLabel, w):
    count = 0
    ans = logisticFunction(np.dot(testData, w.T))
    for item in ans:
        if item[0] > 0.5:
            item[0] = 1
        else: item[0] = 0
    
    for i in range(len(ans)):
        if ans[i][0] == testDataLabel[i][0]:
            count += 1

    return count / len(ans)

def dataProcess(fileName):
    data = []
    trainData, trainDataLabel = [], []
    testData, testDataLabel = [], []

    with open(fileName, "r") as f:
        for line in f:
            if len(line) <= 10:
                continue
            line = line.rstrip("\n").split(",")
            data.append(line)
    # 打乱列表，形成训练集和测试集 
    random.shuffle(data)

    for i in range(len(data)):
        if i / len(data) < 0.63:
            trainData.append(list(map(float, data[i][:4])) + [1])
            if data[i][4] == "Iris-setosa":
                trainDataLabel.append([1])
            else:
                trainDataLabel.append([0])
        else:
            testData.append(list(map(float, data[i][:4])) + [1])
            if data[i][4] == "Iris-setosa":
                testDataLabel.append([1])
            else:
                testDataLabel.append([0])
    # 将list类型转为np.array类型
    trainData = np.array(trainData)
    trainDataLabel = np.array(trainDataLabel)
    testData = np.array(testData)
    testDataLabel = np.array(testDataLabel)

    return(trainData, trainDataLabel, testData, testDataLabel)

if __name__ == "__main__":
    fileName = "iris.data"
    # 数据划分
    trainData, trainDataLabel, testData, testDataLabel = dataProcess(fileName)
    # 函数训练
    w = logisticRegression(trainData, trainDataLabel)
    print(w)
    # 仿真测试
    acc = logisticRegressionTest(testData, testDataLabel, w) 
    print(acc)
