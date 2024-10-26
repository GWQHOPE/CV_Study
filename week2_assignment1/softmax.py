# #Softmax练习
#
# 这个练习类似于SVM练习，以下是学习目标:
#
# - 实现一个完全向量化的 Softmax 分类器的损失函数
# - 实现其**解析梯度**的完全向量化表达式
# - 使用数值梯度**检查实现**
# - 使用验证集来**调整学习率和正则化强度**
# - 使用**随机梯度下降（SGD）优化**损失函数
# - **可视化**最终学习到的权重

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# # get_ipython注释掉是因为，其作用是允许用户访问一些特定于 Notebook 的功能，在python文件中无用会报错
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading extenrnal modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
  """
    从磁盘加载 CIFAR-10 数据集，并执行预处理以准备
    它用于线性分类器。这些步骤与我们用于
    SVM 的步骤相同，但压缩为一个函数。
  """
  # 加载原始的 CIFAR-10 数据
  cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

  # 对数据进行子采样
  # 使用range和np.random.choice选择部分数据
  # 分别生成验证集（1000个样本）、训练集（49000个样本）、测试集（1000个样本）和开发集（500个样本）
  mask = list(range(num_training, num_training + num_validation))
  X_val = X_train[mask]
  y_val = y_train[mask]
  mask = list(range(num_training))
  X_train = X_train[mask]
  y_train = y_train[mask]
  mask = list(range(num_test))
  X_test = X_test[mask]
  y_test = y_test[mask]
  mask = np.random.choice(num_training, num_dev, replace=False)
  X_dev = X_train[mask]
  y_dev = y_train[mask]

  # 预处理：将图像数据重塑为行
  # 将每张图像的维度从 (32, 32, 3) 转换为一维数组
  X_train = np.reshape(X_train, (X_train.shape[0], -1))
  X_val = np.reshape(X_val, (X_val.shape[0], -1))
  X_test = np.reshape(X_test, (X_test.shape[0], -1))
  X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

  # 归一化数据：减去均值图像
  # 通过减去训练集的均值图像来消除偏差，使得训练数据的均值为0。这有助于加速训练过程和提高模型的收敛性。
  mean_image = np.mean(X_train, axis = 0)
  X_train -= mean_image
  X_val -= mean_image
  X_test -= mean_image
  X_dev -= mean_image

  # 添加偏置维度并转换为列
  # 使用np.hstack在每个数据集中添加一个偏置项（全为1的列）
  X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
  X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
  X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
  X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

  # 函数返回处理后的训练集、验证集、测试集和开发集，以及它们的标签。
  return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev


# 调用上述函数以获取我们的数据
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()

# 打印出每个数据集的形状，以确认数据的维度是否符合预期。
print('输出1：各个数据集的形状')
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print('dev data shape: ', X_dev.shape)
print('dev labels shape: ', y_dev.shape)



# softmax分类器
# 此部分所有代码写在softmax.py文件中
# 首先实现带有嵌套循环的朴素softmax损失函数。
# 打开文件 softmax.py 并实现softmax_loss_naive 函数。

# 导入了softmax_loss_naive函数，函数用于计算softmax损失和梯度。
from cs231n.classifiers.softmax import softmax_loss_naive
import time

# 生成一个随机的softmax权重矩阵，并用它来计算损失。
# 生成一个形状为 (3073, 10) 的随机权重矩阵 W，并将其值缩小到一个较小的范围。
# 3037：特征的为维度，10为分类的类别数# 生成一个随机的softmax权重矩阵，并用它来计算损失
W = np.random.randn(3073, 10) * 0.0001
# 调用了之前导入的 softmax_loss_naive 函数
# 使用权重矩阵 W、特征矩阵 X_dev、标签 y_dev 和正则化强度（此处为0.0）来计算softmax损失和对应的梯度
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

# 作为粗略的合理性检查，我们的损失应该接近 -log(0.1)。
print('输出2：softmax损失值')
print('loss: %f' % loss)
print('sanity check: %f' % (-np.log(0.1)))

# ## 相关问题1:
# 我们为什么期望损失接近 -log(0.1)？请简要解释。**
# 
# **答案：** * 由于权重矩阵 W 是均匀随机选择的，因此每个类别的预测概率是均匀分布，并且都等于 1/10，
# 其中 10 是类别的数量。因此每个样本的交叉熵为 -log(0.1)，这应该等于损失。*

# 完成softmax_loss_naive的实现，并实现一个（朴素）版本的梯度，该梯度使用嵌套循环。
# 调用之前实现的 softmax_loss_naive 函数，计算损失值和梯度。
# 这里没有使用正则化项（设置为0.0）
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

# 和SVM一样，使用数值梯度检查作为调试工具。
# 数值梯度应该接近于解析梯度。
print('输出3：梯度检查（数值梯度应该接近于解析梯度）')
# 用于梯度检查的函数 grad_check_sparse。这个函数用于验证计算得到的梯度是否正确。
from cs231n.gradient_check import grad_check_sparse
# 函数f接受权重 w 作为输入，并返回相应的softmax损失（只取第一个返回值，即损失值）
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
# 调用 grad_check_sparse，计算数值梯度并与分析梯度（即通过 softmax_loss_naive 得到的梯度）进行比较，确保两者接近
grad_numerical = grad_check_sparse(f, W, grad, 10)

# 与SVM的情况类似，使用正则化进行另一个梯度检查
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 1e2)
# 计算带有正则化的softmax损失
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 1e2)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)


# 现在我们已经实现了softmax损失函数及其梯度的朴素版本，
# 接下来在softmax_loss_vectorized中实现一个向量化版本。
# 这两个版本应该计算出相同的结果，但向量化版本应该快得多。

# 记录计算的开始时间
tic = time.time()
# 调用朴素实现的 softmax_loss_naive 函数计算损失值 loss_naive 和梯度 grad_naive
# 这里使用的正则化强度为 0.00001
loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.00001)
# 记录计算的结束时间
toc = time.time()
print('输出4：朴素计算VS向量化计算所需时间')
print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))

# 导入向量化的softmax损失函数和梯度计算的实现
from cs231n.classifiers.softmax import softmax_loss_vectorized
tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.00001)
toc = time.time()
# 输出向量化损失值和计算所需时间
print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# 和SVM一样，我们使用Frobenius范数来比较两个版本的梯度。
grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('输出5：朴素计算VS向量化计算损失和梯度比较')
print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
print('Gradient difference: %f' % grad_difference)

# 使用验证集来调整超参数（正则化强度和学习率）。
# 你应该尝试不同范围的学习率和正则化强度；应该能够在验证集上获得超过0.35的分类准确率。

# 导入Softmax分类器的实现
from cs231n.classifiers import Softmax
results = {}
# 初始化最佳验证准确率为负值
best_val = -1
# 存储最佳Softmax分类器的变量
best_softmax = None
# 定义学习率的候选值列表
learning_rates = [1e-7, 2e-7, 5e-7]
#regularization_strengths = [5e4, 1e8]
# 创建正则化强度的候选值列表。该列表包含在范围(1-0.3, 1+0.3)和(5-0.3, 5+0.3)内的值，适用于超参数调整
regularization_strengths =[(1+0.1*i)*1e4 for i in range(-3,4)] + [(5+0.1*i)*1e4 for i in range(-3,4)]

#############################################################################
# TODO:
# 使用验证集来设置学习率和正则化强度。                                      #
# 这应该和你为SVM做的验证是相同的；将最佳训练的softmax分类器保存在best_softmax中。 #
#############################################################################
# 嵌套循环，遍历每个学习率和每个正则化强度
for lr in learning_rates:
    for rs in regularization_strengths:
        softmax = Softmax()# 实例化一个新的Softmax分类器。
        # 用训练数据训练Softmax分类器，指定学习率和正则化强度，训练迭代次数为2000。
        softmax.train(X_train, y_train, lr, rs, num_iters=2000)
        # 计算准确率
        y_train_pred = softmax.predict(X_train)# 使用训练集进行预测
        train_accuracy = np.mean(y_train == y_train_pred)# 计算训练集的准确率
        y_val_pred = softmax.predict(X_val)# 使用验证集进行预测
        val_accuracy = np.mean(y_val == y_val_pred)# 计算验证集的准确率
        # 更新最佳模型
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_softmax = softmax           
        results[(lr,rs)] = train_accuracy, val_accuracy

# 输出结果
print('输出6：不同超参数组合下训练集以及验证集的正确率')
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))

print('输出7：最佳验证集准确率')
print('best validation accuracy achieved during cross-validation: %f' % best_val)

# 在测试集上评估
# 在测试集上评估最佳的Softmax模型

# 使用最佳Softmax模型对测试集进行预测
y_test_pred = best_softmax.predict(X_test)
# 计算测试集的准确率
test_accuracy = np.mean(y_test == y_test_pred)
# 输出测试集的准确率
print('输出8：测试集准确率')
print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))

# 可视化每个类别学习到的权重
w = best_softmax.W[:-1,:]# 去掉偏置项，提取权重
# 将权重重塑为 (32, 32, 3, 10) 形状，适合可视化
w = w.reshape(32, 32, 3, 10)

# 找到权重的最小值和最大值
w_min, w_max = np.min(w), np.max(w)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
  # 创建一个2行5列的子图，并设置当前绘图位置
  plt.subplot(2, 5, i + 1)
  
  # 将权重缩放到0到255之间，首先减去最小值，然后除以范围（最大值减去最小值），最后乘以255进行归一化。
  wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
  # 显示权重图像，将其转换为无符号8位整型
  plt.imshow(wimg.astype('uint8'))
  # 关闭坐标轴，清除边框，以使图像显示更加清晰。
  plt.axis('off')
  # 设置当前子图的标题为对应的类别名称，帮助识别每个图像代表的类别。
  plt.title(classes[i])
plt.show()
print('输出9（图像）：可视化数据集中每个类别学习到的权重')