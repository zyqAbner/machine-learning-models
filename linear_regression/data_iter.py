import numpy as np
import random

def data_iter(batch_size, features, labels):
    """
    小批量数据迭代器
    :param batch_size: 批量大小
    :param features: 特征
    :param labels: 标签
    :return: 返回小批量特征和标签
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]