import numpy as np

def linreg(X, w, b):
    """
    线性回归模型: y = Xw + b
    :param X: 输入特征
    :param w: 权重
    :param b: 偏置
    :return: 模型输出
    """
    return np.dot(X, w) + b