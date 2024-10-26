# # 实现一个神经网络
# 在这个练习中，我们将开发一个由全连接层构成的神经网络，以执行分类任务，并在CIFAR-10数据集上进行测试。
# 导入进行数值计算和绘图的常用工具库
import numpy as np
import matplotlib.pyplot as plt
# 导入TwoLayerNet 类，用于构建和训练一个两层神经网络的类
from cs231n.classifiers.neural_net import TwoLayerNet

# # get_ipython注释掉是因为，其作用是允许用户访问一些特定于 Notebook 的功能，在python文件中无用会报错
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # 设置绘图的默认大小
plt.rcParams['image.interpolation'] = 'nearest' # 设置图像插值方式为最近邻
plt.rcParams['image.cmap'] = 'gray' # 设置图像的色彩映射为灰度

# %load_ext autoreload 和 %autoreload 2 是 IPython 的魔法命令，
# 用于在修改外部模块后自动重新加载模块，方便调试和开发。
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

def rel_error(x, y):
  """ 返回相对误差 """
  # 计算x和y之间的相对误差
  # 避免在计算中出现除以零的情况，确保分母不会小于 1e-8
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# 我们将使用文件`neural_net.py`中的`TwoLayerNet`类来表示我们网络的实例。网络参数存储在实例变量`self.params`中，
# 其中键是字符串形式的参数名称，值是`numpy`数组。下面，我们初始化一些玩具数据和一个玩具模型，用于开发您的实现。
# 创建一个小型神经网络和一些玩具数据来检查你的实现。
# 注意，我们设置了随机种子，以确保实验可重复。

# 输入层的大小
input_size = 4
# 隐藏层的大小
hidden_size = 10
# 类别的数量 表示这是一个多分类问题
num_classes = 3
# 输入样本的数量
num_inputs = 5

def init_toy_model():
    # 设置随机种子为0，以保证结果可重复
    np.random.seed(0)
    # 初始化并返回一个两层神经网络实例
    # 它包含输入层、一个隐藏层和一个输出层，std=1e-1 表示初始化权重时的标准差。
    # 这个网络的输入大小是 input_size，隐藏层的神经元数量是 hidden_size，输出类别数为 num_classes
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    # 设置随机种子为1，确保数据可重复
    np.random.seed(1)
    # 生成形状为(num_inputs, input_size)的输入数据 X，值服从正态分布，并放大 10 倍
    X = 10 * np.random.randn(num_inputs, input_size)
    # 创建一个标签数组 y，包含 5 个标签，值为 0, 1, 2 等
    y = np.array([0, 1, 2, 2, 1])
    # 返回生成的数据和标签
    return X, y

# 初始化神经网络
net = init_toy_model()
# 初始化训练数据和标签
X, y = init_toy_data()


# # 前向传播：计算分数
# 打开文件 `neural_net.py` 查看方法  `TwoLayerNet.loss`.这个函数与在 SVM 和 Softmax 练习中编写的损失函数非常相似：
# 它接受数据和权重，并计算类别分数、损失以及参数的梯度。
# 实现前向传播的第一部分，该部分使用权重和偏置计算所有输入的分数。
# 通过神经网络计算输入数据 X 的分数

scores = net.loss(X)
# 打印你的分数
print('输出1：通过神经网络计算得到的分数')
print('Your scores:')
print(scores)
print()
print('输出2：预先计算好的正确分数')
print('correct scores:')
# 定义数组correct_scores 的数组，包含了所有输入样本的正确分数。
# 这些分数是根据正确的实现预先计算好的，作为基准来验证模型的正确性。
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print(correct_scores)
print()

# 计算你的分数与正确分数之间的差异，应该非常小，期望小于 1e-7
print('输出3：神经网络计算得分与预先计算好的正确分数之差')
print('Difference between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))


# # 前向传播：计算损失
# 在同一个函数中，实现第二部分，用于计算数据损失和正则化损失。
# 计算损失，使用输入数据 X 和真实标签 y，同时设置正则化参数 reg
loss, _ = net.loss(X, y, reg=0.1)

# 正确的损失值
correct_loss = 1.30378789133

# # 差异应该非常小，期望小于 1e-12
print('输出4：计算得到的损失与正确损失值之差')
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))


# # Backward pass
# 实现函数的其余部分。这将计算损失对变量 `W1`, `b1`, `W2`, 和 `b2`的梯度。现在，已经正确实现了前向传播，可以使用数值梯度检查来调试反向传播。
# 导入梯度检查工具

from cs231n.gradient_check import eval_numerical_gradient

# 使用数值梯度检查来验证反向传播的实现。
# 如果实现正确，数值梯度和解析梯度之间的差异
# 对于每个 W1、W2、b1 和 b2 应该小于 1e-8。

# 计算损失和梯度
loss, grads = net.loss(X, y, reg=0.1)
# 传入输入数据 X、真实标签 y 和正则化参数 reg，计算得到当前的损失值 loss 和梯度字典 grads

# 这些差异应该都小于 1e-8 左右
# 遍历每个参数计算相对误差
print('输出5：每个参数（权重、偏置）的最大相对误差')
for param_name in grads:
    # 定义一个匿名函数 f，计算当前参数 W 的损失
    f = lambda W: net.loss(X, y, reg=0.1)[0]
    
    # 计算数值梯度
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    # 打印最大相对误差
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))


# # 训练网络
# 为了训练网络，我们将使用随机梯度下降（SGD），这与支持向量机（SVM）和Softmax分类器相似。查看函数`TwoLayerNet.train`
# 并填写缺失的部分以实现训练过程。这应该与您在SVM和Softmax分类器中使用的训练过程非常相似。
# 还需要实现 `TwoLayerNet.predict`, 因为训练过程中会定期进行预测，以跟踪网络训练过程中的准确性。
# 以旦实现了这些方法，运行下面的代码，以在玩具数据上训练一个两层网络。应该能达到训练损失小于0.2的目标。
# 初始化玩具模型

net = init_toy_model()

# 训练网络，调用 `train` 方法进行训练。
# 传入训练数据 X 和标签 y，以及验证数据 X 和标签 y。
# 使用学习率 1e-1 和正则化强度 1e-5。
# 训练迭代次数为 100 次，verbose=False 表示不打印训练过程中的详细信息
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=1e-5,
            num_iters=100, verbose=False)

# 打印最终的训练损失，`stats['loss_history'][-1]` 获取最后一轮迭代的损失值
print('输出6：打印最终的训练损失')
print('Final training loss: ', stats['loss_history'][-1])

# 绘制损失历史图
# `stats['loss_history']` 包含了每次迭代时的训练损失，使用 `plt.plot()` 来绘制损失随迭代次数的变化。
plt.plot(stats['loss_history'])
# 设置横轴为迭代次数
plt.xlabel('iteration')
# 设置纵轴为训练损失
plt.ylabel('training loss')
# 设置图表标题
plt.title('Training Loss history')
# 显示图表
plt.show()
print('输出7（图像）：损失随迭代次数的变化')

# # 加载数据
# 现在已经实现了一个通过梯度检查并能在玩具数据上正常工作的两层网络，是时候加载我们最喜欢的CIFAR-10数据，
# 以便使用它来训练一个在真实数据集上的分类器。

from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    从磁盘加载CIFAR-10数据集，并执行预处理，以准备将其用于两层神经网络分类器。
    这些步骤与我们为支持向量机（SVM）使用的步骤相同，但简化为一个函数。
    """
    # 指定CIFAR-10数据集的目录
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    # 从指定目录加载CIFAR-10数据，返回训练集和测试集数据及其标签
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
     # 根据给定的数量子样本数据集
    # 生成验证集的掩码（索引范围）
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask] # 从训练集中提取验证集数据
    y_val = y_train[mask] # 从训练集中提取验证集标签
    
     # 生成训练集的掩码（索引范围）
    mask = list(range(num_training))
    X_train = X_train[mask]  # 只保留训练集数据
    y_train = y_train[mask]  # 只保留训练集标签
    
    # 生成测试集的掩码（索引范围）
    mask = list(range(num_test))
    X_test = X_test[mask]  # 只保留测试集数据
    y_test = y_test[mask]  # 只保留测试集标签

    # 数据归一化：减去均值图像，以消除平均偏差
    mean_image = np.mean(X_train, axis=0)  # 计算训练集的均值图像
    X_train -= mean_image  # 从训练集中减去均值图像
    X_val -= mean_image  # 从验证集中减去均值图像
    X_test -= mean_image  # 从测试集中减去均值图像

    # 将数据重塑为行格式，每一行为一个样本
    X_train = X_train.reshape(num_training, -1)  # 重塑训练集
    X_val = X_val.reshape(num_validation, -1)  # 重塑验证集
    X_test = X_test.reshape(num_test, -1)  # 重塑测试集

    # 返回处理后的数据集
    return X_train, y_train, X_val, y_val, X_test, y_test


# 调用上面的函数以获取数据集
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('输出8：各个数据集形状')
# 打印训练数据和标签的形状，以确认数据加载和处理的正确性
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# # 训练网络
# 为了训练我们的网络，我们将使用带动量的随机梯度下降（SGD）。此外，我们将在优化过程中使用指数学习率调度来调整学习率；
# 在每个周期结束后，我们将通过将学习率乘以衰减率来降低学习率。

# 定义一个两层神经网络，使用随机梯度下降法进行训练，并在训练完成后在验证集上评估模型的准确率。

# 定义输入大小、隐藏层大小和类别数量
input_size = 32 * 32 * 3  # CIFAR-10图像的输入大小为32x32像素，3个颜色通道（RGB）
hidden_size = 50          # 隐藏层神经元数量
num_classes = 10          # CIFAR-10有10个类别

# 创建一个两层神经网络实例
net = TwoLayerNet(input_size, hidden_size, num_classes)

# 训练网络
print('输出9：训练过程信息（损失变化）')
stats = net.train(X_train, y_train, X_val, y_val,  # 训练数据及验证数据
                  num_iters=1000,               # 训练迭代次数
                  batch_size=200,               # 每次迭代的批次大小
                  learning_rate=1e-4,           # 初始学习率
                  learning_rate_decay=0.95,     # 学习率衰减率（每个epoch后乘以该值）
                  reg=0.5,                       # L2正则化强度，防止过拟合
                  verbose=True)                  # 是否输出训练过程信息

# 在验证集上进行预测
val_acc = (net.predict(X_val) == y_val).mean()  # 计算预测正确的比例，即验证集的准确率
print('输出10：验证集准确率')
print('Validation accuracy: ', val_acc)  # 输出验证准确率


# # 调试训练
# 使用我们上述提供的默认参数，您应该在验证集上获得约0.29的验证准确率。这并不好。
# 一种获取问题洞察的策略是绘制在优化过程中训练集和验证集上的损失函数及准确率。
# 另一种策略是可视化网络第一层学习到的权重。在大多数训练视觉数据的神经网络中，第一层的权重在可视化时通常会显示出某种可见的结构。


# 可视化训练过程中的损失函数变化和训练集与验证集的分类准确率变化
import matplotlib.pyplot as plt 

# 绘制损失函数和训练/验证准确率
plt.subplot(2, 1, 1)  # 创建一个2行1列的子图，并选择第1个子图
plt.plot(stats['loss_history'])  # 绘制损失历史，损失记录在stats字典中
plt.title('Loss history') 
plt.xlabel('Iteration')  # 设置x轴标签为“迭代次数”
plt.ylabel('Loss')  # 设置y轴标签为“损失”

plt.subplot(2, 1, 2)  # 选择第2个子图
plt.plot(stats['train_acc_history'], label='train')  # 绘制训练集准确率历史
plt.plot(stats['val_acc_history'], label='val')  # 绘制验证集准确率历史
plt.title('Classification accuracy history')  
plt.xlabel('Epoch')  # 设置x轴标签为“周期”
plt.ylabel('Classification accuracy')  # 设置y轴标签为“分类准确率”
plt.legend()  
plt.show() 
print('输出11（图像）：训练过程中的损失函数变化和训练集与验证集的分类准确率变化')
# 导入可视化工具函数 'visualize_grid'，函数用于将多个图片以网格的形式展示
from cs231n.vis_utils import visualize_grid

# 定义一个函数来可视化网络权重

def show_net_weights(net):
  # 获取网络第一层的权重 'W1'，假设网络的参数字典中包含 'W1'（第一层权重）
  W1 = net.params['W1']
  # 重塑权重矩阵 W1 的形状：通过 reshape 将其转换为 (32, 32, 3, num_filters)
  # 32x32 是图片的尺寸，3 是颜色通道（RGB），num_filters 是该层的过滤器数量。
  # 转置矩阵，使得过滤器在最后的维度上，方便按网格显示
  W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
  
  # 将其转换为 uint8 类型以确保显示时颜色正确。
  plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
  # 关闭坐标轴显示，使得展示的图像更清晰，没有多余的轴线
  plt.gca().axis('off')
  plt.show()
print('输出12（图像）：可视化网络权重')
show_net_weights(net)


# # 调整超参数
# 
# **问题所在:**. 从上面的可视化结果来看，我们发现损失函数的下降大致呈线性趋势，这表明学习率可能太低。此外，训练和验证准确率之间没有差距，
# 这表明我们使用的模型容量较低，我们应该增加模型的规模。另一方面，如果模型规模过大，我们预计会看到更多的过拟合，
# 这会表现为训练和验证准确率之间的差距非常大。

# **调优:**. 调整超参数并培养它们对最终性能影响的直觉是使用神经网络的重要部分，因此我们希望多加练习。
# 下面，应该尝试不同的超参数值，包括隐藏层大小、学习率、训练轮数和正则化强度。还可以考虑调整学习率衰减，但使用默认值应该也能获得不错的性能。
# 
# **预期结果:**. 应该力求在验证集上达到超过48%的分类准确率。我们最好的网络在验证集上的准确率超过52%。
# 
# **实验:**: 在本次练习中的目标是尽可能在CIFAR-10数据集上获得良好的结果，
# 使用全连接神经网络。欢迎使用其他技术（例如，使用PCA降维、添加Dropout、或向求解器添加特性等）。

best_net = None  # 将最佳模型存储在这里

#############################################################################
# TODO: 使用验证集调整超参数。将您训练的最佳模型存储在 best_net 中。        #
# 为了帮助调试您的网络，使用类似于我们上面使用的可视化工具可能会有所帮助；      #
# 这些可视化与我们之前看到的调整不当的网络的可视化结果会有显著的定性差异。      #
# 手动调整超参数可能很有趣，但您可能会发现编写代码自动遍历可能的超参数组合      #
# 像我们在之前的练习中一样是很有用的。                                      #
#############################################################################
hidden_size = [75, 100, 125]  # 定义隐藏层大小的候选值

results = {}  # 用于存储每组超参数对应的验证集准确率
best_val_acc = 0  # 初始化最佳验证集准确率
best_net = None  # 初始化最佳网络模型

learning_rates = np.array([0.7, 0.8, 0.9, 1, 1.1]) * 1e-3  # 定义学习率的候选值
regularization_strengths = [0.75, 1, 1.25]  # 定义正则化强度的候选值

print('running', end=' ')  # 输出“正在运行”以指示开始
for hs in hidden_size:  # 遍历每个隐藏层大小
    for lr in learning_rates:  # 遍历每个学习率
        for reg in regularization_strengths:  # 遍历每个正则化强度
            print('.', end=' ')  # 每次迭代输出一个点以指示进度
            net = TwoLayerNet(input_size, hs, num_classes)  # 创建一个两层神经网络
            
            # 训练网络
            stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1500, batch_size=200,
            learning_rate=lr, learning_rate_decay=0.95,
            reg=reg, verbose=False)  # 训练网络，传入超参数
            
            # 计算验证集上的准确率
            val_acc = (net.predict(X_val) == y_val).mean()  # 预测并计算准确率
            
            # 如果当前验证集准确率超过最佳准确率，则更新最佳准确率和模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc  # 更新最佳验证集准确率
                best_net = net  # 更新最佳网络模型
            
            # 存储当前超参数组合的验证集准确率
            results[(hs, lr, reg)] = val_acc  

print()  # 输出换行
print("finished")  # 输出“完成”

# 打印结果。
print('输出13：不同超参数组合下验证集正确率')
for hs, lr, reg in sorted(results):  # 遍历所有超参数组合的结果
    val_acc = results[(hs, lr, reg)]  # 获取当前组合的验证集准确率
    print('hs %d lr %e reg %e val accuracy: %f' % (hs, lr, reg, val_acc))  # 打印超参数组合和对应的验证集准确率
    
# 打印在交叉验证期间取得的最佳验证集准确率
print('输出14：最佳验证集正确率')
print('best validation accuracy achieved during cross-validation: %f' % best_val_acc)

# 可视化最佳网络的权重
print('输出15（图像）：可视化最佳网络的权重')
show_net_weights(best_net) # 调用函数显示最佳模型的权重


# # 在测试集上运行
# 
# 当完成实验后，应在测试集上评估最终训练的网络；此时应该获得超过 48% 的准确率。

# 使用最佳网络在测试集上进行预测并计算测试准确率 对测试集 `X_test` 进行预测，返回预测结果
# 将预测结果与真实标签 `y_test` 进行比较，生成布尔数组，表示每个预测是否正确
test_acc = (best_net.predict(X_test) == y_test).mean()
# 打印测试准确率  Python3中使用 print(test_acc)
print('输出16：测试集准确率')
print('Test accuracy: ', test_acc)