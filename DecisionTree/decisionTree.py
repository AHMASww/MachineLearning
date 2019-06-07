'''
该算法还缺少预剪枝和后剪枝，可能存在过拟合的现象，导致泛化性能较差
'''

import numpy as np
import random, math
from collections import defaultdict, deque

# 定义树结构
class treeNode():
    def __init__(self, val, dimension):
        self.val = val
        self.dimension = dimension
        self.left = None
        self.right = None

# 原始数据处理
def dataProcess(fileName):
    data = []
    trainData, testData = [], []
    with open(fileName, "r") as f:
        for line in f:
            line.rstrip("\n")
            line = line.split(" ")
            line = list(map(float, line))
            data.append(line)
    random.shuffle(data)
    nums = len(data)
    demarcation = int(nums * 0.632)
    trainData = data[:demarcation]
    data = data[demarcation:]
    verifyData = data[:len(data)//2]
    testData = data[len(data)//2:]


    return(trainData, verifyData, testData)

# 计算香农熵找出最优子属性，返回值为某一属性的最优子属性，此处因为训练集是随机划分的，故最优子属性每次训练不唯一
def calcShannonEnt(dataSet, dimension):
    # 初始化值 
    vals_dimension = []
    keys_dimension = []
    ent0, ent1 = 0, 0
    min_ent = float("inf")
    min_ent_val = float("inf")

    for data in dataSet:
        if data[dimension-1] not in vals_dimension:
            vals_dimension.append(data[dimension-1])
    vals_dimension = sorted(vals_dimension)
    for i in range(len(vals_dimension)-1):
        keys_dimension.append((vals_dimension[i] + vals_dimension[i+1]) / 2)

    # print(keys_dimension)
    for key in keys_dimension:
        temp_data = defaultdict(list)
        # 根据连续值划分方法划分数据,其结果只可能是二分的
        for data in dataSet:
            if data[dimension-1] <= key:
                temp_data[0].append(data)
            else:
                temp_data[1].append(data)
        # 将划分过的数据计算Ent
        ent0 = shannonEnt(temp_data[0])
        ent1 = shannonEnt(temp_data[1])
        # 计算权重下的Ent(ID3)
        sum_ent = len(temp_data[0]) / len(dataSet) * ent0 + len(temp_data[1]) / len(dataSet) * ent1
        # 计算C4.5
        t0 = len(temp_data[0]) / len(dataSet) * math.log2(len(temp_data[0]) / len(dataSet))
        t1 = len(temp_data[1]) / len(dataSet) * math.log2(len(temp_data[1]) / len(dataSet))
        sum_ent /= -(t0 + t1)
        # print(key, sum_ent)
        # 求出该维度下最好的连续值划分值
        if sum_ent < min_ent:
            min_ent = sum_ent
            min_ent_val = key

    return (min_ent, min_ent_val, dimension)

def shannonEnt(dataSet):
    dic = defaultdict(int)
    sum_ent = 0

    for data in dataSet:
        dic[data[-1]] += 1
    
    for i in dic:
        sum_ent -= dic[i] / len(dataSet) * math.log2(dic[i] / len(dataSet))

    return sum_ent

# 终止条件1，判断数据集是否为空或者当前数据均属于一类，返回值为bool类型
def termination1(dataSet):
    # 标签列表存储数据的标签，超过两个则放回False，表示不能继续分类
    label = []

    # 判断数据是否为空，是则返回False，表示不能继续分类
    if len(dataSet) == 0:
        return False

    for item in dataSet:
        if item[-1] not in label and len(label) == 0:
            label.append(item[-1])
        # 超过两种不同标识的数据，可以进一步分类
        elif item[-1] not in label and len(label) == 1:
            return True 
    # label中只有一种标识，不能继续分类
    return False

# 终止条件2，判断属性集是否为空，或者当前数据集在该属性集上相同，无法根据决策树算法进一步分类，返回值为bool类型
def termination2(dataSet, dimensionSet):
    if not dimensionSet: return False
    for i in range(len(dataSet)-1):
        # 数据集并不完全一致，可以继续分类 
        for j in dimensionSet:
            if dataSet[i][j-1] != dataSet[i+1][j-1]:
                return True 
    # 数据集完全一致，无法继续分类
    return False 

# 决策树算法(ID3)
# 这里主要是对离散数据进行决策树算法
def decisionTree(dataSet, dimensionSet):
    # 原始数据处理，得到训练集和测试集
    # 对训练集进行决策树算法：
    #   判断终止条件：
    #       1.当前数据集是否为空或者所有均属于一类
    #       2.属性集为空，或者当前数据集在该属性集上相同
    #   寻找最优子属性对数据集进行划分
    #   以上循环，直至全部到了终止条件
    if termination1(dataSet) == False or termination2(dataSet, dimensionSet) == False:
        dic = defaultdict(int)
        for item in dataSet:
            dic[item[-1]] += 1
        if len(dic) == 0:
            return None
        else:
            if dic[0] > dic[1]:
                return treeNode(0, None)
            else: return treeNode(1, None)
    # 存储每个维度的最优子属性值和计算出的ent值
    # print(dimensionSet)
    curNode = treeNode(None, None)

    bestDimension = []
    for dimension in dimensionSet:
        bestDimension.append(calcShannonEnt(dataSet, dimension))
    # 获取整体最优子属性
    bestDimension = sorted(bestDimension, key = lambda x : x[0])[0]
    curNode.val = bestDimension[1]
    curNode.dimension = bestDimension[2]

    leftDataSet, rightDataSet = [], []
    # print(bestDimension)
    for item in dataSet:
        if item[bestDimension[2]-1] <= bestDimension[1]:
            leftDataSet.append(item)
        else: rightDataSet.append(item)
    # print(len(leftDataSet))
    # print(len(rightDataSet))

    curNode.left = decisionTree(leftDataSet, dimensionSet - {bestDimension[2]})
    curNode.right = decisionTree(rightDataSet, dimensionSet - {bestDimension[2]})

    return curNode

# 决策时算法验证
def testDecisionTree(dataSet, root):
    correct = 0

    for item in dataSet:
        temp_root = root
        while temp_root.dimension != None:
            if item[temp_root.dimension - 1] <= temp_root.val:
                temp_root = temp_root.left
            else: temp_root = temp_root.right
        if temp_root.val == item[-1]:
            correct += 1
    
    return (correct / len(dataSet))

# 输出决策树
def printDecisionTree(root):
    queue = deque()
    queue.append(root)
    while True:
        temp = []
        while queue:
            cur_node = queue.popleft()
            print(cur_node.val, cur_node.dimension, end = " ")
            print("  ", end = " ")
            if cur_node.left:
                temp.append(cur_node.left)
            if cur_node.right:
                temp.append(cur_node.right)
        print("")
        if not temp: break
        else: queue.extend(temp)

# 主函数
# 执行决策树算法
if __name__ == "__main__":
    fileName = "data.txt"
    cor = 0
    for i in range(20):
        # 划分训练集,验证集和测试集
        trainData, verifyData, testData = dataProcess(fileName)
        # 获取数据集的属性维度
        dimensionSet = {i for i in range(1, len(trainData[0]))}
        # 建立决策树，返回根节点 
        root = decisionTree(trainData, dimensionSet)
        # 输出决策树
        # printDecisionTree(root)
        # 测试集验证
        cor += testDecisionTree(testData, root)
    print(cor/20)