import numpy as np
from model import linreg
from loss import squared_loss
from optimizer import sgd
from data import synthetic_data
from data_iter import data_iter
import matplotlib.pyplot as plt

def train(lr, num_epoch, batch_size):
    # 初始化模型参数
    w = np.random.normal(0, 0.01, (2, 1))
    b = np.zeros(1)
    w_grad = np.zeros_like(w)
    b_grad = np.zeros_like(b)

    # 生成数据集
    true_w = np.array([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 用来记录损失的列表
    train_losses = []

    # 开始训练
    for epoch in range(num_epoch):
        for X, y in data_iter(batch_size, features, labels):
            # 正向传播和损失计算
            y_hat = linreg(X, w, b)

            # 反向传播：计算梯度
            batch_size = len(X)            
            w_grad = np.dot(X.T, (y_hat - y)) / batch_size
            b_grad = np.sum(y_hat - y) / batch_size

            # 更新参数
            w -= lr * w_grad
            b -= lr * b_grad
        # 计算训练集上的损失
        train_loss = squared_loss(linreg(features, w, b), labels)
        train_losses.append(np.mean(train_loss))

        print(f"Epoch {epoch + 1}, Loss: {np.mean(train_loss)}")
    
    # 可视化训练过程中的损失变化
    plt.plot(range(num_epoch), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

    return w, b