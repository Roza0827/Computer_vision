import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

# 导入Fashion MNIST数据集
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
out = np.max(y_train)

# 转换格式
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)
# 独热编码
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


class NeuralNetwork:

    def __init__(self, hidden_size, activation, regularization_strength=0.1):
        # 权重和偏置的初始化

        self.W1 = np.random.randn(X_train.shape[1], hidden_size) / np.sqrt(X_train.shape[1])
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, out + 1) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, out + 1))

        # 激活函数
        self.activation = activation

        # 正则化强度
        self.regularization_strength = regularization_strength

   
    # 激活函数及其导数
    def _activation(self, x, derivative=False):
        if self.activation == 'relu':
            if derivative:
                return np.where(x <= 0, 0, 1)
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            if derivative:
                return self._activation(x) * (1 - self._activation(x))
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'softmax':  # 添加 softmax 激活函数
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 避免指数溢出
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    # 反向传播
    def _backprop(self, X, y):
        # 前向传播
        z1 = X @ self.W1 + self.b1
        a1 = self._activation(z1)
        z2 = a1 @ self.W2 + self.b2
        y_pred = self._activation(z2, derivative=False)  # 使用 softmax 激活函数
        # y = y.astype(np.int32)
        # y_one_hot = tf.keras.utils.to_categorical(y, num_classes=10)
        # 计算梯度
        dL_dy_pred = y_pred - y  # 将y转换为独热编码形式
        dL_dz2 = dL_dy_pred * self._activation(z2, derivative=True)
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0)
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * self._activation(z1, derivative=True)
        dL_dW1 = X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0)

        # 正则化
        dL_dW1 += self.regularization_strength * self.W1
        dL_dW2 += self.regularization_strength * self.W2

        # 返回梯度
        return dL_dW1, dL_db1, dL_dW2, dL_db2

    # 训练
    def train(self, X, y, batch_size=64, num_epochs=20, learning_rate=0.01):
        # 初始化损失记录
        train_losses = []
        test_losses = []
        test_accuracies = []

        # 划分批次
        batches = [(X[i:i + batch_size], y[i:i + batch_size]) for i in range(0, len(X), batch_size)]

        # 训练
        for epoch in range(num_epochs):
            # 学习率衰减
            learning_rate = learning_rate * 0.95

            # 遍历批次
            for batch_X, batch_y in batches:
                # 反向传播
                dL_dW1, dL_db1, dL_dW2, dL_db2 = self._backprop(batch_X, batch_y)

                # 更新权重和偏置
                self.W1 -= learning_rate * dL_dW1
                self.b1 -= learning_rate * dL_db1
                self.W2 -= learning_rate * dL_dW2
                self.b2 -= learning_rate * dL_db2

            # 计算损失和准确率
            # y = tf.keras.utils.to_categorical(y, num_classes=10)
            train_loss = self.loss(X, y)
            train_losses.append(train_loss)
            test_loss, test_accuracy = self.evaluate(X_test, y_test)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

        # 返回损失记录
        return train_losses, test_losses, test_accuracies

    # 损失函数
    def loss(self, X, y):
        z1 = X @ self.W1 + self.b1
        a1 = self._activation(z1)
        z2 = a1 @ self.W2 + self.b2
        y_pred = self._activation(z2)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # 评估
    def evaluate(self, X, y):
        # y1 = y
        # y = tf.keras.utils.to_categorical(y, num_classes=10)
        z1 = X @ self.W1 + self.b1
        a1 = self._activation(z1)
        z2 = a1 @ self.W2 + self.b2
        y_pred = self._activation(z2)
        # print(y_pred.shape)
        # print(y.shape)
        return self.loss(X, y), np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))


# 模型超参数
hidden_sizes = [32, 64, 128]
activations = ['sigmoid']
regularization_strengths = [0.01, 0.001, 0.0001]

# 参数查找
best_accuracy=0
for hidden_size in hidden_sizes:
    for activation in activations:
        for regularization_strength in regularization_strengths:
            # 训练神经网络
            model = NeuralNetwork(hidden_size=hidden_size, activation=activation,
                                  regularization_strength=regularization_strength)
            train_losses, test_losses, test_accuracies= model.train(X_train, y_train, num_epochs=10)
            validation_accuracy = test_accuracies[-1]
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_model = model
            # 记录结果
            print(
                f'Hidden size: {hidden_size}, Activation: {activation}, Regularization strength: {regularization_strength}')
            print(f'Validation accuracy: {test_accuracies[-1]}')
np.save('best_model_weights.npy', {'W1': best_model.W1, 'b1': best_model.b1, 'W2': best_model.W2, 'b2': best_model.b2})


# 评估最佳模型
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# 可视化训练过程
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train loss')
plt.plot(test_losses, label='Test loss')
plt.plot(test_accuracies, label='Test accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.title('Training process')
plt.show()



