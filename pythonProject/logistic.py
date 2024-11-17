import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# 读取 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理：扁平化输入和归一化
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# 将标签转换为独热编码
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 设置权重和偏置
w = tf.Variable(tf.zeros([784, 10]), name="w")
b = tf.Variable(tf.zeros([10]), name="b")


# 定义逻辑回归模型
def logistic_regression(x):
    logits = tf.matmul(x, w) + b
    return logits


# 定义损失函数
def compute_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))


# 定义准确率计算
def compute_accuracy(y_true, y_pred):
    correct_predictions = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


# 设置优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# 训练逻辑
loss_history = []
epochs = 50
batch_size = 100
num_batches = len(x_train) // batch_size

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(num_batches):
        # 获取当前批次数据
        start = i * batch_size
        end = start + batch_size
        x_batch = x_train[start:end]
        y_batch = y_train[start:end]

        # 前向传播和梯度计算
        with tf.GradientTape() as tape:
            logits = logistic_regression(x_batch)
            loss = compute_loss(y_batch, logits)

        # 反向传播并更新参数
        gradients = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradients, [w, b]))
        epoch_loss += loss.numpy()

    # 记录平均损失
    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# 测试集准确率
logits_test = logistic_regression(x_test)
accuracy = compute_accuracy(y_test, logits_test).numpy()
print(f"Test Accuracy: {accuracy:.4f}")

# 可视化损失变化
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.show()
