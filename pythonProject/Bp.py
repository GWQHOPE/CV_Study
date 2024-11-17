import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

# 加载 MNIST 数据集
(data_train, labels_train), (data_test, labels_test) = mnist.load_data()
data_train = data_train.reshape(-1, 784).astype("float32") / 255.0
data_test = data_test.reshape(-1, 784).astype("float32") / 255.0
labels_train = tf.keras.utils.to_categorical(labels_train, 10)
labels_test = tf.keras.utils.to_categorical(labels_test, 10)

# 设置超参数
n_input = 784  # 输入特征数
n_labels = 10  # 输出类别数
n_hidden_layer = 30  # 隐藏层神经元数量
max_epochs = 10000  # 最大迭代次数
batch_size = 100  # 批量大小
learning_rate = 0.2  # 学习率
seed = 0  # 随机种子
np.random.seed(seed)
tf.random.set_seed(seed)

# 定义 Sigmoid 函数及其导数
def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 初始化权重和偏置
weights = {
    "w_1": tf.Variable(tf.random.normal([n_input, n_hidden_layer], seed=seed)),
    "w_2": tf.Variable(tf.random.normal([n_hidden_layer, n_labels], seed=seed)),
}
biases = {
    "b_1": tf.Variable(tf.random.normal([n_hidden_layer], seed=seed)),
    "b_2": tf.Variable(tf.random.normal([n_labels], seed=seed)),
}

# 构建模型
def forward_pass(x):
    h_1 = tf.matmul(x, weights["w_1"]) + biases["b_1"]
    o_1 = sigmoid(h_1)
    h_2 = tf.matmul(o_1, weights["w_2"]) + biases["b_2"]
    o_2 = sigmoid(h_2)
    return h_1, o_1, h_2, o_2

# 定义训练函数
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        # 前向传播
        h_1, o_1, h_2, y_hat = forward_pass(x_batch)
        # 计算损失
        loss = tf.reduce_mean(tf.square(y_hat - y_batch))  # 均方误差损失
        # 反向传播：计算梯度
    gradients = tape.gradient(loss, list(weights.values()) + list(biases.values()))
    # 更新权重和偏置
    optimizer.apply_gradients(zip(gradients, list(weights.values()) + list(biases.values())))
    return loss

# 定义准确率计算
def compute_accuracy(x, y):
    _, _, _, y_hat = forward_pass(x)
    correct_predictions = tf.equal(tf.argmax(y_hat, axis=1), tf.argmax(y, axis=1))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# 初始化优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# 数据分批
train_dataset = tf.data.Dataset.from_tensor_slices((data_train, labels_train)).batch(batch_size)

# 开始训练
for epoch in range(max_epochs):
    for x_batch, y_batch in train_dataset:
        train_loss = train_step(x_batch, y_batch)

    # 每 1000 次迭代打印一次训练和测试准确率
    if epoch % 1000 == 0:
        train_acc = compute_accuracy(data_train, labels_train)
        test_acc = compute_accuracy(data_test, labels_test)
        print(
            "Epoch: {0}, Train Accuracy: {1:.4f}, Test Accuracy: {2:.4f}".format(
                epoch, train_acc.numpy(), test_acc.numpy()
            )
        )
