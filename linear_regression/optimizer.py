import numpy as np

def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降
    :param params: 模型参数 (w, b)
    :param lr: 学习率
    :param batch_size: 批量大小
    """
    for param in params:
        param[:] = param - lr * param.grad / batch_size