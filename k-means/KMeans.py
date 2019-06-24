import numpy as np
from collections import defaultdict 

def dataProcess(fileName):
    data, label = [], []
    with open(fileName, "r") as f:
        for line in f:
            if line == "\n":
                continue
            else:
                line = line.rstrip("\n").split(",")
                data.append(list(map(float, line[:4])))
                label.append([line[-1]])
    data = np.array(data)
    label = np.array(label)
    return data, label

def init(data):
    k = defaultdict(np.ndarray)
    k[0] = data[0]
    k[1] = data[50]
    k[2] = data[100]
    return k

# k is dict
def k_means(data, k, times, newClass, oldClass):
    for t in range(times):
        for i in range(len(data)):
            min_distance = float("inf")
            min_c = -1
            for c in k:
                distance = np.sqrt(np.sum(np.square(data[i] - k[c])))
                if distance < min_distance:
                    min_distance = distance
                    min_c = c
            newClass.append(min_c)
        
        if newClass == oldClass:
            return newClass
        else:
            oldClass = newClass
            newClass = []

        new_k = defaultdict(np.ndarray)
        count = defaultdict(int)
        for i in range(len(oldClass)):
            if oldClass[i] not in new_k:
                new_k[oldClass[i]] = 0
            new_k[oldClass[i]] += data[i]
            count[oldClass[i]] += 1
        for i in new_k:
            new_k[i] = new_k[i] / count[i]
        k = new_k
    
    return oldClass 

if __name__ == "__main__":
    fileName = "iris.data"
    data, label = dataProcess(fileName)
    k = init(data)
    classifer = k_means(data, k, 20, [], [])
    print(classifer)