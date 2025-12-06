import numpy as np

def synthetic_data(w, b, num_examples):
    """
    生成线性回归数据: y = Xw + b + 噪声
    :param w: 权重
    :param b: 偏置
    :param num_examples: 样本数
    :return: 生成的数据集 (X, y)
    """
    X = np.random.normal(0, 1, (num_examples, len(w))) # 生成特征
    y = np.dot(X, w) + b # 生成标签
    y += np.random.normal(0, 0.01, y.shape) # 加入噪声
    return X, y.reshape((-1, 1))