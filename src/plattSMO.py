import numpy as np
from svm import *
import matplotlib.pyplot as plt
import time


class PlattSMO:
    def __init__(self, dataMat, classlabels, C, toler, maxIter, **kernelargs):
        self.x = array(dataMat)
        self.label = array(classlabels).transpose()
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        self.m = shape(dataMat)[0]
        self.n = shape(dataMat)[1]
        self.alpha = array(zeros(self.m), dtype='float64')
        self.b = 0.0
        self.eCache = array(zeros((self.m, 2)))
        self.K = zeros((self.m, self.m), dtype='float64')
        self.kwargs = kernelargs
        self.SV = ()
        self.SVIndex = None
        for i in range(self.m):
            for j in range(self.m):
                self.K[i, j] = self.kernelTrans(self.x[i, :], self.x[j, :])

    def calcEK(self, k):
        fxk = dot(self.alpha * self.label, self.K[:, k]) + self.b
        Ek = fxk - float(self.label[k])
        return Ek

    def updateEK(self, k):
        Ek = self.calcEK(k)

        self.eCache[k] = [1, Ek]

    def selectJ(self, i, Ei):
        maxE = 0.0
        selectJ = 0
        Ej = 0.0
        validECacheList = nonzero(self.eCache[:, 0])[0]
        if len(validECacheList) > 1:
            for k in validECacheList:
                if k == i: continue
                Ek = self.calcEK(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxE:
                    selectJ = k
                    maxE = deltaE
                    Ej = Ek
            return selectJ, Ej
        else:
            selectJ = selectJrand(i, self.m)
            Ej = self.calcEK(selectJ)
            return selectJ, Ej

    def innerL(self, i):
        Ei = self.calcEK(i)
        if (self.label[i] * Ei < -self.toler and self.alpha[i] < self.C) or \
                (self.label[i] * Ei > self.toler and self.alpha[i] > 0):
            self.updateEK(i)
            j, Ej = self.selectJ(i, Ei)
            alphaIOld = self.alpha[i].copy()
            alphaJOld = self.alpha[j].copy()
            if self.label[i] != self.label[j]:
                L = max(0, self.alpha[j] - self.alpha[i])
                H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
            else:
                L = max(0, self.alpha[j] + self.alpha[i] - self.C)
                H = min(self.C, self.alpha[i] + self.alpha[j])
            if L == H:
                return 0
            eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
            if eta >= 0:
                return 0
            self.alpha[j] -= self.label[j] * (Ei - Ej) / eta
            self.alpha[j] = clipAlpha(self.alpha[j], H, L)
            self.updateEK(j)
            if abs(alphaJOld - self.alpha[j]) < 0.00001:
                return 0
            self.alpha[i] += self.label[i] * self.label[j] * (alphaJOld - self.alpha[j])
            self.updateEK(i)
            b1 = self.b - Ei - self.label[i] * self.K[i, i] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[i, j] * (self.alpha[j] - alphaJOld)
            b2 = self.b - Ej - self.label[i] * self.K[i, j] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[j, j] * (self.alpha[j] - alphaJOld)
            if 0 < self.alpha[i] and self.alpha[i] < self.C:
                self.b = b1
            elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def smoP(self):
        iter = 0
        entrySet = True
        alphaPairChanged = 0
        while iter < self.maxIter and ((alphaPairChanged > 0) or (entrySet)):
            alphaPairChanged = 0
            if entrySet:
                for i in range(self.m):
                    alphaPairChanged += self.innerL(i)
                iter += 1
            else:
                nonBounds = nonzero((self.alpha > 0) * (self.alpha < self.C))[0]
                for i in nonBounds:
                    alphaPairChanged += self.innerL(i)
                iter += 1
            if entrySet:
                entrySet = False
            elif alphaPairChanged == 0:
                entrySet = True
        self.SVIndex = nonzero(self.alpha)[0]
        self.SV = self.x[self.SVIndex]
        self.SVAlpha = self.alpha[self.SVIndex]
        self.SVLabel = self.label[self.SVIndex]
        self.x = None
        self.K = None
        self.label = None
        self.alpha = None
        self.eCache = None

    #   def K(self,i,j):
    #       return self.x[i,:]*self.x[j,:].T
    def kernelTrans(self, x, z):
        if array(x).ndim != 1 or array(x).ndim != 1:
            raise Exception("input vector is not 1 dim")
        if self.kwargs['name'] == 'linear':
            return sum(x * z)
        elif self.kwargs['name'] == 'rbf':
            theta = self.kwargs['theta']
            return exp(sum((x - z) * (x - z)) / (-1 * theta ** 2))

    # def calcw(self):
    #     for i in range(self.m):
    #         self.K += dot(self.alpha[i] * self.label[i], self.x[i, :])

    def predict(self, testData):
        test = array(testData)
        # return (test * self.w + self.b).getA()
        result = []
        m = shape(test)[0]
        for i in range(m):
            tmp = self.b
            for j in range(len(self.SVIndex)):
                tmp += self.SVAlpha[j] * self.SVLabel[j] * self.kernelTrans(self.SV[j], test[i, :])
            while tmp == 0:
                tmp = random.uniform(-1, 1)
            if tmp > 0:
                tmp = 1
            else:
                tmp = -1
            result.append(tmp)
        return result


def plotDecisionBoundary(data, label, SV, smo, b, alphas):
    # 创建网格点
    x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
    y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # 获取预测结果
    Z = np.array(smo.predict(np.c_[xx.ravel(), yy.ravel()]))

    # 绘制决策边界
    Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # 绘制数据点
    plt.scatter(data[:, 0], data[:, 1], c=label, cmap=plt.cm.Paired)

    # 绘制支持向量
    for sv in SV:
        plt.scatter(sv[0], sv[1], s=100, facecolors='none', edgecolors='pink', marker='o', alpha=0.5)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Decision Boundary with Support Vectors')
    plt.show()


def plotDataWithHyperplane(data, label, SV, b, alphas):
    n = shape(data)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for i in range(n):
        if int(label[i]) == 1:
            x1.append(data[i][0])
            y1.append(data[i][1])
        else:
            x2.append(data[i][0])
            y2.append(data[i][1])
    ax.scatter(x1, y1, s=10, c='red', marker='s', label='Class 1')
    ax.scatter(x2, y2, s=10, c='green', marker='s', label='Class -1')

    # Plot support vectors
    for sv in SV:
        ax.scatter(sv[0], sv[1], s=100, facecolors='none', edgecolors='blue', marker='o', alpha=0.5)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('Data Points with Support Vectors')
    plt.show()


def performance_measure(dataMat, labelMat):
    start_time = time.time()
    smo = PlattSMO(dataMat, labelMat, 6, 0.0001, 10000, name='rbf', theta=1.3)
    smo.smoP()
    end_time = time.time()
    # plotDataWithHyperplane(dataMat, labelMat, smo.SV, smo.b, smo.alpha)
    # plotDecisionBoundary(np.array(dataMat), np.array(labelMat), smo.SV, smo, smo.b, smo.alpha)
    testResult = smo.predict(dataMat)
    m = shape(dataMat)[0]
    count = 0.0
    for i in range(m):
        if labelMat[i] != testResult[i]:
            count += 1
    errorRate = count / m
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


if __name__ == "__main__":
    datasets = ['SmallDataSet.txt', 'MediumDataSet.txt', 'LargeDataSet.txt']
    sizes = [100, 500, 1000]
    execution_times = []
    error_rates = []
    for dataset in datasets:
        dataMat, labelMat = loadDataSet(dataset)
        execution_time, error_rate = performance_measure(dataMat, labelMat)
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
