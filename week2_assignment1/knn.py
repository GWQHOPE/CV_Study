# # KNN练习
# kNN分类器由两个阶段组成:
# - 在训练过程中，分类器获取训练数据并简单地记住它
# - 在测试过程中，kNN通过比较所有训练图像并转移k个最相似训练示例的标签来对每个测试图像进行分类
# - k的值经过交叉验证
# 在本练习中，会实现这些步骤，了解基本的图像分类管道、交叉验证，并熟练编写高效的矢量化代码。

import random
import numpy as np
#  从 CS231n 的数据工具库中导入 load_CIFAR10 函数，用于加载 CIFAR-10 数据集
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# # get_ipython注释掉是因为，其作用是允许用户访问一些特定于 Notebook 的功能，在python文件中无用会报错
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # 定义绘图的默认大小为 10 x 8 英寸
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 这是一个 IPython 的魔法命令，用于加载 autoreload 扩展，以便在代码更改后自动重新加载模块。
# get_ipython().run_line_magic('load_ext', 'autoreload')
#  设置 autoreload 的模式为 2，这意味着每次执行代码时，都会自动重新加载所有模块
#  （除了在启动时已经加载的模块）
# get_ipython().run_line_magic('autoreload', '2')

# 接下来将加载 CIFAR-10 数据集
# 定义 CIFAR-10 数据集的路径。
# 这个路径是相对于当前工作目录的，指向存放 CIFAR-10 数据集的文件夹。
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# 用于输出训练和测试数据的形状，确保数据加载正确
print('输出1：训练数据训练标签形状以及测试数据测试标签形状')
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
# 输出结果：表示训练集有 50000 张 32x32 像素的 RGB 图像，测试集有 10000 张相同规格的图像。
# 从 CIFAR-10 数据集中可视化不同类别的样本图像
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)

# 定义每个类别要显示的样本数量，这里设为 7。
samples_per_class = 7
# 使用 enumerate 函数遍历 classes 列表，y 表示类别索引，cls 表示类别名称
for y, cls in enumerate(classes):
    # 找出训练标签中等于当前类别索引 y 的所有样本的索引，返回一个一维数组 idxs
    idxs = np.flatnonzero(y_train == y)
    # 随机不重复选择7个样本索引
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    # 遍历选中的样本索引 idxs，计算绘图的位置索引，结合i与y定位当前图像在子图中的位置
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        # 显示选定样本的图像，将图像数据转换为无符号整数格式，确保图像正确显示。
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        # 隐藏图像的坐标轴，使图像清晰可见
        if i == 0:
            plt.title(cls)
plt.show()
print('输出2（图像）：可视化不同类别的样本图像')

# 接下来的代码将对数据进行子采样，以提高执行效率
# 定义训练集中要使用的样本数量
num_training = 5000
# 创建索引列表，采样保留标签。
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

#  定义测试集中要使用的样本数量，这里设定为 500。同样设置索引然后采样保留标签。
num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]


# 接下来的代码将图像数据重新调整为行的格式
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
# 输出重塑后的训练集和测试集的形状。这将帮助我们确认数据是否正确重塑。
print('输出3：重塑后数据集形状')
print(X_train.shape, X_test.shape)
# 输出中3072=32*32*3 32*32像素，有3个颜色通道

# 从 cs231n.classifiers 模块中导入 KNearestNeighbor 类
from cs231n.classifiers import KNearestNeighbor

# 接下来将创建一个 kNN 分类器实例
# kNN 分类器的过程实际上并不会进行任何复杂的计算或学习。
# kNN 分类器仅仅是存储训练数据，后续的分类决策是通过对这些数据进行距离计算实现的
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)


# 我们现在想用kNN分类器对测试数据进行分类。回想一下，我们可以将这个过程分为两个步骤：
# 
# 1.首先，我们必须计算所有测试示例和所有训练示例之间的距离。
# 2.给定这些距离，对于每个测试示例，我们找到k个最近的示例，并让它们投票给标签。
# 
# 
# 让我们从计算所有训练和测试示例之间的距离矩阵开始。例如，如果有**Ntr**训练示例和**Nte**测试示例，
# 则此阶段应产生**Nte x Ntr**矩阵，
# 其中每个元素（i，j）是第i个测试和第j个训练示例之间的距离。
# 
# 打开 `cs231n/classifiers/k_nearest_neighbor.py`并实现函数`compute_distances_two_loops`，
# 该函数在所有（测试、训练）示例对上使用（非常低效）双环，并一次计算一个元素的距离矩阵。


# 打开 k_nearest_neighbor.py 文件，并在其中实现 compute_distances_two_loops 方法。
# 该方法的目的是计算测试样本与训练样本之间的距离。

# 测试实现
dists = classifier.compute_distances_two_loops(X_test)
print('输出4：距离矩阵的形状')
print(dists.shape)

# 我们可以可视化距离矩阵：每一行都是一个测试示例
# 显示生成的距离矩阵图像
plt.imshow(dists, interpolation='none')
print('输出5（图像）：距离矩阵图像')
plt.show()


# **相关问题1：** 注意距离矩阵中的结构化图案，其中一些行或列更亮。（请注意，在默认配色方案中，黑色表示低距离，白色表示高距离。）
# 
# - 数据中明显明亮的行背后的原因是什么？
# - 是什么导致了这些列？

# **答案**: 
# *如果第i个测试数据与大量列车数据相似，则第i行将为黑色。否则，第i行将是白色的-如果第j列数据与大量测试数据相似，
# 则第j列将为黑色。否则，第j列将是白色的。

# 利用最近邻算法（k=1）进行分类预测，并计算分类的准确率
y_test_pred = classifier.predict_labels(dists, k=1)

# 计算并打印正确预测示例的分数
# 计算正确预测的数量。y_test_pred是模型的预测结果，而y_test是实际的标签。
num_correct = np.sum(y_test_pred == y_test)
# 计算准确率
accuracy = float(num_correct) / num_test
print('输出6：模型预测的准确率')
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

# 应该看到大约“27%”的准确率。现在让我们尝试一个更大的`k`，比如`k=5`：
# 设置k为5
y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('输出7：增大k值后模型预测的准确率')
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

# 应该期望看到比k=1时稍好的性能。
# 现在让我们通过使用部分向量化和一个循环来加速距离矩阵的计算。
# 实现函数 compute_distances_one_loop 并运行下面的代码。
dists_one = classifier.compute_distances_one_loop(X_test)

# 为了确保我们的向量化实现是正确的，我们需要确保它与朴素实现一致。
# 有很多方法可以判断两个矩阵是否相似；最简单的方法之一是 Frobenius 范数。
# 弗罗贝尼乌斯范数: 该范数计算两个矩阵之间的差异，通过计算所有元素差的平方和的平方根来实现。
# 如果你之前没有见过，两个矩阵的 Frobenius 范数是所有元素差异的平方和的平方根；
# 换句话说，将矩阵重塑为向量，并计算它们之间的欧几里得距离。
difference = np.linalg.norm(dists - dists_one, ord='fro')  
# 计算两个距离矩阵的 Frobenius 范数
print('输出7：两个距离矩阵（朴素实现VS向量化）的差异')
print('Difference was: %f' % (difference, ))  # 输出差异
if difference < 0.001:
  print('Good! The distance matrices are the same')  
else:
  print('Uh-oh! The distance matrices are different')  

# 现在在 compute_distances_no_loops 中实现完全向量化的版本并运行代码
dists_two = classifier.compute_distances_no_loops(X_test)

# 检查距离矩阵是否与之前计算的相同：
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('输出8：两个距离矩阵（朴素实现VS完全向量化）的差异')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
  print('Good! The distance matrices are the same')
else:
  print('Uh-oh! The distance matrices are different')

# 比较不同实现的运行速度
def time_function(f, *args):
  """
   调用函数 f，并传入参数 args，返回执行所需的时间（以秒为单位）。
  """
  import time
  tic = time.time()
  f(*args)
  toc = time.time()
  return toc - tic
# 计算不同实现的执行时间
print('输出9：不同距离计算方法的执行速度（双重循环、单循环、向量化计算）')
two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print('Two loop version took %f seconds' % two_loop_time)

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print('One loop version took %f seconds' % one_loop_time)

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)

# 无循环版本的时间显著低于其他两种实现，验证了向量化的优势。


# ### 交叉验证
# 
# 我们已经实现了 k-最近邻分类器，但我们将 k 的值设为 5 是随意的。现在，我们将通过交叉验证来确定这个超参数的最佳值。

num_folds = 5  # 设置折数为5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]  # k的选择列表

X_train_folds = []  # 用于存储训练数据折叠的列表
y_train_folds = []  # 用于存储训练标签折叠的列表
############################################################################
# TODO:                                                                       
# 将训练数据分割成折叠。分割后，X_train_folds和y_train_folds应都是长度为num_folds的
# 列表，其中y_train_folds[i]是X_train_folds[i]中点的标签向量。                   #
# 提示：查阅numpy的array_split函数。                                         #
#############################################################################
# pass
y_train_ = y_train.reshape(-1, 1)  # 将y_train重塑为列向量
X_train_folds, y_train_folds = np.array_split(X_train, 5), np.array_split(y_train_, 5)
# 将X_train和y_train分成5个折叠

# 一个字典，用于存储我们在交叉验证时找到的不同k值的准确率。
# 在运行交叉验证后，k_to_accuracies[k]应是长度为num_folds的列表，给出在使用该k值时找到的不同准确率值。
k_to_accuracies = {}

#############################################################################
# TODO:                                                                       
# 执行k折交叉验证以找到最佳的k值。对于每一个可能的k，运行k最近邻算法num_folds次，  #
# 在每种情况下，使用所有但一个的折作为训练数据，最后一个折作为验证集。将所有折叠和所有   #
# k值的准确率存储在k_to_accuracies字典中。                                 #
#############################################################################
for k_ in k_choices:
    k_to_accuracies.setdefault(k_, [])  # 初始化字典，确保每个k都有一个列表
for i in range(num_folds):
    classifier = KNearestNeighbor()  # 初始化k最近邻分类器
    X_val_train = np.vstack(X_train_folds[0:i] + X_train_folds[i+1:]) 
    # 合并除第i个折外的所有折作为训练数据
    y_val_train = np.vstack(y_train_folds[0:i] + y_train_folds[i+1:]) 
    # 合并除第i个折外的所有标签
    y_val_train = y_val_train[:, 0]  
    # 将标签重塑为一维数组
    classifier.train(X_val_train, y_val_train)  # 训练分类器
    for k_ in k_choices:
        y_val_pred = classifier.predict(X_train_folds[i], k=k_) 
        # 使用第i个折作为验证集进行预测
        num_correct = np.sum(y_val_pred == y_train_folds[i][:, 0])  
        # 计算正确预测的数量
        accuracy = float(num_correct) / len(y_val_pred)  # 计算准确率
        k_to_accuracies[k_] = k_to_accuracies[k_] + [accuracy]  
        # 将准确率添加到对应k值的列表中

# 打印出计算出的准确率
print('输出10：不同k值对应的准确率')
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))  # 打印每个k值的准确率

# 通过这样的交叉验证，可以有效地选择出最适合当前数据集的 k 值，从而提升 k-NN 分类器的性能。

# 绘制原始观测值
for k in k_choices:
  accuracies = k_to_accuracies[k]
  plt.scatter([k] * len(accuracies), accuracies)

# 计算并绘制趋势线和误差条：
# 计算每个 k 值对应的准确率的平均值和标准差
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
# 各 k 值的准确率均值
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
# 各 k 值准确率的标准差，反映了准确率的波动程度
# 使用 plt.errorbar 函数绘制带有误差条的趋势线
# 趋势线：模型总体表现如何随k值的变化而变化，判断最优k
# 误差条：不同k值下准确率的标准差（模型稳定性的体现）
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()
# 散点图：对于每个k，有多个准确率测量结果，模型有波动性
print('输出11（图像）：不同的k值下的交叉验证准确率')
# 根据上面的交叉验证结果，选择最佳的 k 值，  
# 使用所有训练数据重新训练分类器，并在测试数据上进行测试。
# 你应该能够在测试数据上获得超过 28% 的准确率。
best_k = 10

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)  # 使用训练数据训练分类器
y_test_pred = classifier.predict(X_test, k=best_k)  
# 在测试数据上进行预测，使用最佳的 k 值

# 计算并显示准确率
num_correct = np.sum(y_test_pred == y_test)  # 计算正确预测的数量
accuracy = float(num_correct) / num_test  # 计算准确率
print('输出12：测试集的预测结果，准确率')
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))  # 输出结果