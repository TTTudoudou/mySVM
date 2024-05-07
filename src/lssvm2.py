from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import time


def kernelTrans(X, A, kTup):
    X = mat(X)
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.K = mat(zeros((self.m, self.m)))  # 特征数据集合中向量两两核函数值组成的矩阵，[i,j]表示第i个向量与第j个向量的核函数值
        self.SV = ()
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def leastSquares(dataMatIn, classLabels, C, kTup):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, kTup)
    # 1.参数设置
    unit = mat(ones((oS.m, 1)))  # [1,1,...,1].T
    I = eye(oS.m)
    zero = mat(zeros((1, 1)))
    upmat = hstack((zero, unit.T))
    downmat = hstack((unit, oS.K + I / float(C)))
    # 2.方程求解
    completemat = vstack((upmat, downmat))  # lssvm中求解方程的左边矩阵
    rightmat = vstack((zero, oS.labelMat))  # lssvm中求解方程的右边矩阵
    b_alpha = completemat.I * rightmat
    oS.b = b_alpha[0, 0]
    for i in range(oS.m):
        oS.alphas[i, 0] = b_alpha[i + 1, 0]
    return oS.alphas, oS.b, oS.K


def predict(alphas, b, dataMat, testVec, kTup):
    Kx = kernelTrans(dataMat, testVec, kTup)  # 可以对alphas进行稀疏处理找到更准确的值
    predict_value = Kx.T * alphas + b
    return sign(float(predict_value))


def plotDecisionBoundary(data, label, alphas, b, kTup):
    # 构造网格
    x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
    y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            testVec = [xx[i, j], yy[i, j]]
            Z[i, j] = predict(alphas, b, data, testVec,kTup)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    n = shape(data)[0]
    dataArr = np.array(data)
    labelArr = np.array(label)
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for i in range(n):
        if int(labelArr[i]) == 1:
            x1.append(data[i][0])
            y1.append(data[i][1])
        else:
            x2.append(data[i][0])
            y2.append(data[i][1])

    plt.scatter(x1, y1, s=10, c='red', marker='s', label='Class 1')
    plt.scatter(x2, y2, s=10, c='green', marker='s', label='Class -1')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Data Points with Support Vectors')
    plt.show()


def performance_measure(dataMat, labelMat, C, kTup):
    start_time = time.time()
    alphas, b, K = leastSquares(dataMat, labelMat, C, kTup)
    end_time = time.time()
    #plotVectors(np.array(dataMat), np.array(labelMat), alphas)
    #plotDecisionBoundary(np.array(dataMat), np.array(labelMat), alphas, b, K)
    error = 0.0
    for i in range(len(dataMat)):
        test = predict(alphas, b, dataMat, dataMat[i],kTup)
        if test != float(labelMat[i]):
            error += 1.0
    errorRate = error / len(dataMat)
    return end_time - start_time, errorRate


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


if __name__ == '__main__':
    sizes = [100, 500, 1000]
    datasets = ['SmallDataSet.txt', 'MediumDataSet.txt', 'LargeDataSet.txt']
    C = 0.6
    k1 = 0.3
    kernel = 'rbf'
    kTup = (kernel, k1)
    execution_times = []
    error_rates = []
    for dataset in datasets:
        dataMat, labelMat = loadDataSet(dataset)
        execution_time, error_rate = performance_measure(dataMat, labelMat, C, kTup)
        execution_times.append(execution_time)
        error_rates.append(error_rate)
        print("数据集：", dataset, "    执行时间：", execution_time, "    错误率：", error_rate)

    # 绘制性能指标随数据集大小的变化趋势
    plt.plot(sizes, execution_times, label="Execution Time")
    plt.xlabel("Dataset Size")
    plt.ylabel("Execution Time (s)")
    plt.title("Performance Trend: Execution Time vs. Dataset Size")
    plt.legend()
    plt.show()

    plt.plot(sizes, error_rates, label="Error Rate")
    plt.xlabel("Dataset Size")
    plt.ylabel("Error Rate")
    plt.title("Performance Trend: Error Rate vs. Dataset Size")
    plt.legend()
    plt.show()
