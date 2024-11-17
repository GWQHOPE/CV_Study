import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # 标准化
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
train_labels = tf.keras.utils.to_categorical(train_labels, 10)  # One-hot 编码
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# 参数初始化
input_num = 784  # 输入的列数
labels = 10      # 输出的类别数
batchsize = 128  # 每批次数据量
max_epochs = 10  # 迭代次数
learning_rate = 0.01  # 学习率

# 数据输入占位符
x = tf.keras.Input(shape=(28, 28, 1))
y = tf.keras.Input(shape=(10,))

# 构建卷积神经网络
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# 创建模型
model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 模型训练
history = model.fit(train_images, train_labels,
                    validation_data=(test_images, test_labels),
                    epochs=max_epochs, batch_size=batchsize, verbose=1)

# 绘制训练曲线
plt.figure(figsize=(12, 5))

# 绘制准确率
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 绘制损失值
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
