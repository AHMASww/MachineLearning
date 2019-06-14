import random
import math
from collections import defaultdict

# 处理原始数据
def dataProcess(fileName):
    with open(fileName, "r") as f:
        data = []
        label = []
        trainData, trainDataLabel, testData, testDataLabel = [], [], [], []
        for line in f:
            if not line or line == "\n": continue
            line = line.rstrip("\n")
            line = line.split(",")
            data.append(line[1:])

    random.shuffle(data)
    length = len(data)
    for i in range(length):
        if i / length < 2 / 3:
            trainData.append(data[i][:-1])
            trainDataLabel.append(data[i][-1])
        else:
            testData.append(data[i][:-1])
            testDataLabel.append(data[i][-1])
            
    return trainData, trainDataLabel, testData, testDataLabel

# 朴素贝叶斯算法
def naiveBayesClassier(trainData, trainDataLabel, testData, testDataLabel):
    # 这里处理连续属性采用密度函数是正态分布
    correct = 0
    # 好西瓜，坏西瓜个数
    goodWatermelon, badWatermelon = 0, 0
    # 对训练集进行预处理
    dic = defaultdict(dict)
    dic["是"] = defaultdict(int) 
    dic["否"] = defaultdict(int)
    Lgood, Lbad = [[], []], [[], []]
    for i in range(len(trainData)):
        if trainDataLabel[i] == "是": 
            goodWatermelon += 1
            Lgood[0].append(trainData[i][6])
            Lgood[1].append(trainData[i][7])
        else: 
            badWatermelon += 1
            Lbad[0].append(trainData[i][6])
            Lbad[1].append(trainData[i][7])
        for j in range(6):
            dic[trainDataLabel[i]][trainData[i][j]] += 1
    # 计算出均值和方差
    U1good, U1bad, Var1good, Var1bad = calUAndVar(Lgood[0], Lbad[0])
    U2good, U2bad, Var2good, Var2bad = calUAndVar(Lgood[1], Lbad[1])
    # 对测试集数据逐个判断
    for i in range(len(testData)):
        Pgood = goodWatermelon / len(trainData)
        Pbad = badWatermelon / len(trainData)
        for j in range(len(testData[0])):
            if j < 6:
                Pgood *= dic["是"][testData[i][j]] / goodWatermelon
                Pbad *= dic["否"][testData[i][j]] / badWatermelon
            Pgood *= standardNormal(U1good, Var1good, float(testData[i][6]))
            Pbad *= standardNormal(U1bad, Var1bad, float(testData[i][6]))
            Pgood *= standardNormal(U2good, Var2good, float(testData[i][7]))
            Pbad *= standardNormal(U2bad, Var2bad, float(testData[i][7]))
        if Pgood >= Pbad and testDataLabel[i] == "是":
            correct += 1
        elif Pgood < Pbad and testDataLabel[i] == "否":
            correct += 1
    # 计算准确率，并返回
    return correct / len(testData)

# 计算均值和方差
def calUAndVar(good, bad):
    Ugood, Ubad, Vargood, Varbad = 0, 0, 0, 0
    for i in range(len(good)):
        Ugood += float(good[i])
    for i in range(len(bad)):
        Ubad += float(bad[i])
    Ugood /= len(good)
    Ubad /= len(bad)
    for i in range(len(good)):
        Vargood += (float(good[i]) - Ugood) ** 2
    for i in range(len(bad)):
        Varbad += (float(bad[i]) - Ubad) ** 2
    Vargood /= len(good)
    Varbad /= len(bad)
    return Ugood, Ubad, Vargood, Varbad

# 正态分布
def standardNormal(u, var, x):
    return (1.0 / (math.sqrt(2 * math.pi * var))) * math.exp(-(x - u) ** 2 / (2 * var))

# 主函数
if __name__ == "__main__":
    fileName = "data.txt"
    # 获取训练集和测试集
    trainData, trainDataLabel, testData, testDataLabel = dataProcess(fileName)
    # 朴素贝叶斯算法
    accuracy = naiveBayesClassier(trainData, trainDataLabel, testData, testDataLabel)
    print(accuracy)