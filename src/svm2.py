import numpy as np
import matplotlib.pyplot as plt
import time


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, gamma=0.1):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.alpha = None
        self.b = None
        self.gamma = gamma
        self.X = None
        self.y = None
        self.support_vectors = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X = X
        self.y = y

        # 计算 Gram 矩阵
        K = self._rbf_kernel(X, X)

        # 初始化 alpha 参数
        self.alpha = np.zeros(n_samples)

        # 初始化阈值 b
        self.b = 0

        for _ in range(self.n_iters):
            for idx in range(n_samples):
                condition = y[idx] * (np.sum(self.alpha * y * K[idx]) - self.b) >= 1
                if condition:
                    self.alpha[idx] -= self.lr * (2 * self.lambda_param * self.alpha[idx])
                else:
                    self.alpha[idx] -= self.lr * (2 * self.lambda_param * self.alpha[idx] - y[idx])
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        K = self._rbf_kernel(X, self.X)
        return np.sign(np.sum(self.alpha * self.y * K, axis=1) - self.b)

    def _rbf_kernel(self, X1, X2):
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                K[i, j] = np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        return K

def plotDecisionBoundary(dataMat, labelMat,svm_classifier):
    # 绘制训练数据点
    plt.scatter(np.array(dataMat)[:, 0], np.array(dataMat)[:, 1], c=np.array(labelMat), cmap=plt.cm.Paired)
    # 绘制决策边界
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.show()


def performance_measure(dataMat, labelMat):
    start_time = time.time()
    # 创建 SVM 分类器实例
    svm_classifier = SVM(gamma=0.1)
    # 拟合模型
    svm_classifier.fit(np.array(dataMat), np.array(labelMat))
    end_time = time.time()
    #plotDecisionBoundary(dataMat, labelMat, svm_classifier)

    error = 0.0
    test = svm_classifier.predict(np.array(dataMat))
    for i in range(len(dataMat)):
        if test[i] != float(labelMat[i]):
            error += 1.0
    errorRate = error / len(dataMat)
    return end_time - start_time, errorRate


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





