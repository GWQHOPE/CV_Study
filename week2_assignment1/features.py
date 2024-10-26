# 图像特征练习

# 我们已经看到，通过在输入图像的像素上训练线性分类器，我们可以在图像分类任务中取得合理的性能。
# 在这个练习中，我们将展示通过训练线性分类器使用从原始像素计算的特征而不是直接使用原始像素，可以提高我们的分类性能。

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# # get_ipython注释掉是因为，其作用是允许用户访问一些特定于 Notebook 的功能，在python文件中无用会报错
# get_ipython().run_line_magic('matplotlib', 'inline')
# 这行代码允许 Matplotlib 在 Jupyter Notebook 中直接显示图形，而不是在单独的窗口中弹出
plt.rcParams['figure.figsize'] = (10.0, 8.0) # 设置图形的默认大小
plt.rcParams['image.interpolation'] = 'nearest' # 设置图像插值方法为最近邻插值
plt.rcParams['image.cmap'] = 'gray' # 设置图像的颜色映射为灰度

# 一些额外的魔法，使 notebook 自动重新加载外部 Python 模块；在python文件中注释掉
# 这两行代码通过 IPython 扩展加载了自动重新加载模块的功能，使得在 notebook 中修改外部 Python 模块后，无需手动重新加载，便于开发和调试。
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

# ## 加载数据
# 与之前的练习类似，我们将从磁盘加载 CIFAR-10 数据。
# 加载 CIFAR-10 数据集，数据集包含图像和相应的标签。
from cs231n.features import color_histogram_hsv, hog_feature
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
  # 加载原始的 CIFAR-10 数据
  cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
  
  # 对数据进行子采样
  mask = list(range(num_training, num_training + num_validation))  
  # 创建验证集的掩码，从训练集中选取验证数据
  X_val = X_train[mask]  
  y_val = y_train[mask]  
  # 根据掩码选取验证集的图像和标签
 
  mask = list(range(num_training))  
  # 创建训练集的掩码，包含从 0 到 num_training-1 的索引
  X_train = X_train[mask]  
  y_train = y_train[mask]  
  # 根据掩码更新训练集的图像和标签

  mask = list(range(num_test))  
  # 创建测试集的掩码，包含从 0 到 num_test-1 的索引
  X_test = X_test[mask]  
  y_test = y_test[mask]  
  # 根据掩码更新测试集的图像和标签

  return X_train, y_train, X_val, y_val, X_test, y_test  
  # 返回训练集、验证集和测试集的图像及标签

# 调用函数以获取 CIFAR-10 数据
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()  


# ## 特征提取
# 对于每张图像，我们将计算 方向梯度直方图（HOG）以及使用 HSV 颜色空间中的色调通道计算的
# 颜色直方图。我们通过将 HOG 特征向量和颜色直方图特征向量拼接起来，
# 形成每张图像的最终特征向量。
# 
# 大致来说，HOG 特征应当捕捉图像的纹理信息，而忽略颜色信息；
# 而颜色直方图则表示输入图像的颜色信息，而忽略纹理。由此，
# 我们期望将两者结合使用能够比单独使用其中任何一个特征效果更好。在额外任务部分验证这一假设是一个不错的尝试。
# `hog_feature` 和 `color_histogram_hsv` 函数都作用于单一图像，并返回该图像的特征向量。`extract_features` 函数接受一组图像和一个特征
# 函数列表，并对每张图像评估每个特征函数，将结果存储在一个矩阵中，其中每一列是单一图像所有特征向量的拼接。
from cs231n.features import *

num_color_bins = 10  # 颜色直方图中的箱数
# 定义特征函数列表，其中包括 HOG 特征和 HSV 颜色空间的颜色直方图特征
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
# 提取训练集的特征
print('输出1：提取训练集特征的进度')
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
# verbose=True的设置会输出提取特征的进度
# 提取验证集的特征
X_val_feats = extract_features(X_val, feature_fns)
# 提取测试集的特征
X_test_feats = extract_features(X_test, feature_fns)

# 预处理：减去均值特征
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)  # 计算训练集特征的均值
X_train_feats -= mean_feat  # 从训练集特征中减去均值
X_val_feats -= mean_feat  # 从验证集特征中减去均值
X_test_feats -= mean_feat  # 从测试集特征中减去均值

# 预处理：除以标准差。这确保每个特征的尺度大致相同。
std_feat = np.std(X_train_feats, axis=0, keepdims=True)  # 计算训练集特征的标准差
X_train_feats /= std_feat  # 将训练集特征除以标准差
X_val_feats /= std_feat  # 将验证集特征除以标准差
X_test_feats /= std_feat  # 将测试集特征除以标准差

# 预处理：添加偏置维度
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])  # 将训练集特征和偏置项拼接
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
# 将验证集特征和偏置项拼接
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))]) 
# 将测试集特征和偏置项拼接


# ## 在特征上训练支持向量机（SVM）
# 
# 使用之前作业中开发的多类别 SVM 代码，在提取的特征上训练 SVM；这应该会比直接在原始像素上训练 SVM 取得更好的结果。


from cs231n.classifiers.linear_classifier import LinearSVM

# learning_rates = [1e-9, 1e-8, 1e-7]  # 原学习率选项
# regularization_strengths = [1e5, 1e6, 1e7]  # 原正则化强度选项
learning_rates =[5e-9, 7.5e-9, 1e-8]  # 选择新的学习率
regularization_strengths = [(5+i)*1e6 for i in range(-3, 4)]  # 生成正则化强度范围

results = {}  # 存储每对参数的训练和验证精度
best_val = -1  # 最佳验证精度初始化
best_svm = None  # 保存最佳的 SVM 模型

#pass
##############################################################################
# TODO:                                                                       
# 使用验证集来设置学习率和正则化强度。                                     #
# 这应该与之前为 SVM 进行的验证相同；将最佳训练的分类器保存到 best_svm 中。   #
# 可能还想尝试不同数量的颜色直方图的箱数。应该能够在验证集上获得接近 0.44 的准确率。       
#############################################################################
for rs in regularization_strengths:  # 遍历所有正则化强度
    for lr in learning_rates:  # 遍历所有学习率
        svm = LinearSVM()  # 初始化线性 SVM
        # 训练 SVM 模型，使用特征数据、标签、学习率、正则化强度和迭代次数
        loss_hist = svm.train(X_train_feats, y_train, lr, rs, num_iters=6000)
        # 在训练集上进行预测
        y_train_pred = svm.predict(X_train_feats)
        # 计算训练精度
        train_accuracy = np.mean(y_train == y_train_pred)
        # 在验证集上进行预测
        y_val_pred = svm.predict(X_val_feats)
        # 计算验证精度
        val_accuracy = np.mean(y_val == y_val_pred)
        # 如果当前验证精度比最佳验证精度更好，则更新最佳验证精度和最佳 SVM
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_svm = svm           
        # 将当前学习率和正则化强度的训练精度和验证精度存储在结果中
        results[(lr, rs)] = train_accuracy, val_accuracy
#pass

# 输出结果
print('输出2：SVM模型下学习率和正则化强度排序后按序输出对应的训练精度和验证精度')
for lr, reg in sorted(results):  # 按学习率和正则化强度排序结果
    train_accuracy, val_accuracy = results[(lr, reg)]  # 获取训练和验证精度
    # 打印当前学习率、正则化强度、训练精度和验证精度
    print(('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy)))
    
# 打印在交叉验证过程中获得的最佳验证精度
print('输出3：使用SVM模型下交叉验证获得的最佳验证精度')
print(('best validation accuracy achieved during cross-validation: %f' % best_val))


# 在测试集上评估训练好的 SVM 模型
# 使用最佳 SVM 模型对测试集特征进行预测
y_test_pred = best_svm.predict(X_test_feats) 
# 计算测试集的准确率，比较真实标签与预测结果
# 这里使用了布尔数组的均值计算，返回的值是正确预测的比例。
test_accuracy = np.mean(y_test == y_test_pred)
# 打印测试集的准确率
print('输出4：SVM模型下测试集准确率')
print(test_accuracy)

# 了解算法如何工作的一个重要方式是可视化它所犯的错误。
# 在这个可视化中，我们展示了当前系统错误分类的图像示例。
# 第一列显示的是系统标记为“飞机”的图像，但其真实标签是其他类别。

examples_per_class = 8  # 每个类别展示的示例数量
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # 类别列表

# 遍历每个类别及其索引
for cls, cls_name in enumerate(classes):
    # 找到所有真实标签不是当前类别但被错误预测为当前类别的索引
    idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
    # 随机选择指定数量的错误分类示例，确保不重复
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    
    # 对每个随机选择的索引进行遍历
    for i, idx in enumerate(idxs):
        # 在子图中绘制图像，排列方式为每列一个类别，每行展示多个示例
        plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
        plt.imshow(X_test[idx].astype('uint8'))  
        plt.axis('off')  
        
        # 仅在第一行显示类别标题
        if i == 0:
            plt.title(cls_name)

# 展示所有图像
print('输出5（图像）：输出错误预测的图像，可视化算法错误')
plt.show()


# ### 相关问题1:
# 描述你所看到的误分类结果。这些结果合理吗？
# 
# 由于我们使用了颜色直方图特征和HOG特征，对于一些具有特殊背景或轮廓的误分类结果，它们是合理的。
# 例如，具有蓝色背景的物体往往会被误分类为飞机，而一些狗（卡车）可能被误分类为猫（汽车）。

# ## 在图像特征上训练神经网络
# 在这次作业中，我们看到在原始像素上训练一个两层神经网络的分类性能优于线性分类器在原始像素上的表现。
# 在本笔记中，我们看到基于图像特征的线性分类器的表现优于线性分类器在原始像素上的表现。
# 
# 为了完整性，我们还应该尝试在图像特征上训练一个神经网络。这种方法应该能够超越之前的所有方法：
# 应该能够轻松在测试集上实现超过55%的分类准确率；我们的最佳模型的分类准确率约为60%。

# 输出训练样本的数量和每个样本的特征数量
print('输出6：训练样本数量及样本的特征数量')
print(X_train_feats.shape)

from cs231n.classifiers.neural_net import TwoLayerNet

# 输入特征的维度
input_dim = X_train_feats.shape[1]  
# 隐藏层的神经元数量
hidden_dim = 500  
# 分类的数量
num_classes = 10  

# 创建一个两层神经网络的实例
net = TwoLayerNet(input_dim, hidden_dim, num_classes)
best_net = None  # 用于存储表现最好的模型

############################################################################
# TODO: 在图像特征上训练一个两层神经网络。可能需要像之前的部分一样进行交叉验证各种参数。 #
# 将最好的模型存储在 best_net 变量中。                                   #
############################################################################

results = {}  # 存储每个超参数组合的结果
best_val = -1  # 记录最佳验证准确率
best_net = None  # 记录表现最好的网络

# 设置不同的学习率
learning_rates = [1e-2, 1e-1, 5e-1, 1, 5]
# 设置不同的正则化强度
regularization_strengths = [1e-3, 5e-3, 1e-2, 1e-1, 0.5, 1]

# 遍历所有的学习率和正则化强度
for lr in learning_rates:
    for reg in regularization_strengths:
        # 为每一组超参数创建一个新的神经网络实例
        net = TwoLayerNet(input_dim, hidden_dim, num_classes)  
        
        # 训练网络
        stats = net.train(X_train_feats, y_train, X_val_feats, y_val,
                          num_iters=1500,  # 训练迭代次数
                          batch_size=200,  # 每个批次的样本数
                          learning_rate=lr,  # 当前学习率
                          learning_rate_decay=0.95,  # 学习率衰减因子
                          reg=reg,  # 当前正则化强度
                          verbose=False)  # 是否打印训练过程信息
                          
        # 计算在验证集上的准确率
        val_acc = (net.predict(X_val_feats) == y_val).mean()  
        
        # 更新最佳值和最佳模型
        if val_acc > best_val:
            best_val = val_acc
            best_net = net  
        
        # 将当前超参数组合的验证准确率存储到结果字典中
        results[(lr, reg)] = val_acc

# 输出结果
print('输出7：两层神经网络模型下按序输出学习率正则化强度组合对应的验证准确率')
for lr, reg in sorted(results):
    val_acc = results[(lr, reg)]
    print(('lr %e reg %e val accuracy: %f' % (
                lr, reg, val_acc)))  # 打印学习率、正则化强度及验证准确率
print('输出8：两层神经网络模型下最佳验证准确率')
print(('best validation accuracy achieved during cross-validation: %f' % best_val))  
# 打印交叉验证期间获得的最佳验证准确率
#pass

# 在测试集上运行你的神经网络分类器。应该能够获得超过55%的准确率。

# 调用最佳网络（best_net）的预测方法，对测试特征（X_test_feats）进行预测。
test_acc = (best_net.predict(X_test_feats) == y_test).mean()
print('输出9：神经网络分类器下测试集准确率')
print(test_acc) # 输出准确率


# 额外任务：设计你自己的特征！
# 
# 你已经看到简单的图像特征可以提高分类性能。到目前为止，
# 我们尝试了HOG（方向梯度直方图）和颜色直方图，但其他类型的特征可能能够实现更好的分类性能。
