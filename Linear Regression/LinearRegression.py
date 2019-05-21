import numpy as np

def lossFunctions(x, w, y):
    fx = np.dot(x, w.T)
    loss = 0.5 * np.mean(np.square(fx - y))
    return loss

def linearRegressionTrain(x, y):
    w = np.ones((1, np.shape(x)[1]))
    learningRate = 0.01 
    times = 10000

    for i in range(times):
        loss = lossFunctions(x, w, y)
        if i % 100 == 0:
            print(w)
            print(loss)
        temp = []
        for i in range(np.shape(w)[1]):
            if i != np.shape(w)[1] - 1:
                t = np.mean(np.dot(np.dot(x, w.T) - y, x[:, i].reshape(1, len(x))))
            else:
                t = np.mean(np.dot(x, w.T) - y)
            temp.append([t])
        w -= learningRate * np.array(temp).T
    return w

def linearRegressionTest(w, testData, testDataLabel):
    print(lossFunctions(testData, w, testDataLabel))

def dataProcess(fileName):
    orginData = []
    trainData, trainDataLabel = [], []
    testData, testDataLabel = [], []

    with open(fileName, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            line = list(map(float, line.split(" ")))
            orginData.append(line + [1])

    nums = len(orginData)
    for i in range(nums):
        if i < nums * 2 // 3:
            trainDataLabel.append([orginData[i][0]])
            trainData.append(orginData[i][1:])
        else:
            testDataLabel.append([orginData[i][0]])
            testData.append(orginData[i][1:])

    return(trainData, trainDataLabel, testData, testDataLabel)

if __name__ == "__main__":
    fileName = "data.txt"
    # 数据划分
    trainData, trainDataLabel, testData, testDataLabel = dataProcess(fileName)
    # 将训练集，训练集标签，测试集，测试集标签转为np.array类型
    trainData = np.array(trainData)
    trainDataLabel = np.array(trainDataLabel)
    testData = np.array(testData)
    testDataLabel = np.array(testDataLabel)
    # 对数据预处理：特征缩放
    trainData = (trainData - np.min(trainData)) / (np.max(trainData) - np.min(trainData))
    testData = (testData - np.min(testData)) / (np.max(testData) - np.min(testData))
    # 训练集开始训练
    w = linearRegressionTrain(trainData, trainDataLabel)
    # 测试集误差
    linearRegressionTest(w, testData, testDataLabel)