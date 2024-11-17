import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 创建卷积神经网络模型
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(5, 5), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 打印模型结构
model.summary()

# 设置模型保存路径
checkpoint_dir = './ckpt/'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:02d}.ckpt')

# 创建回调函数，保存每个 epoch 的模型权重
save_model_cb = ModelCheckpoint(filepath=checkpoint_path,
                                save_weights_only=True,
                                verbose=1,
                                save_freq='epoch')  # 每个 epoch 保存一次

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels,
          epochs=10,
          batch_size=128,
          validation_split=0.1,
          callbacks=[save_model_cb])

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\n测试集的准确率：')
print("准确率: %.4f，共测试了%d张图片" % (test_acc, len(test_images)))
