import numpy as np

def squared_loss(y_hat, y):
    """
    均方损失函数: L(y_hat, y) = (y_hat - y)^2 / 2
    :param y_hat: 预测值
    :param y: 真实值
    :return: 损失值
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2