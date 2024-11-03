# # Dropout 随机失活
# Dropout [1] 是一种通过在前向传播中随机将一些特征设置为零来正则化神经网络的技术。
# 在本练习中，你将实现一个 dropout 层，并修改全连接网络以可选使用 dropout。
# #
# # [1] Geoffrey E. Hinton 等，"通过防止特征检测器的共同适应来改善神经网络"，arXiv 2012

import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
def rel_error(x, y):
  """ 返回相对误差 """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# 加载（预处理过的）CIFAR10 数据
data = get_CIFAR10_data()
print('输出1：训练集、验证集以及测试集形状')
for k, v in data.items():
  print('%s: ' % k, v.shape)


# # Dropout 前向传播
# # 在文件 `cs231n/layers.py` 中实现 dropout 的前向传播。由于 dropout 在训练和测试时表现不同，请确保为两种模式实现操作。
# #
# # 完成后运行下面的单元格以测试你的实现。
x = np.random.randn(500, 500) + 10  # 随机生成一个 500x500 的矩阵
print('输出2：不同置零概率设置下下训练模式和测试模式下的输出均值和置零比例')
for p in [0.3, 0.6, 0.75]:
  out, _ = dropout_forward(x, {'mode': 'train', 'p': p}) # 训练模式的前向传播
  out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p}) # 测试模式的前向传播

  print('Running tests with p = ', p)
  print('Mean of input: ', x.mean()) # 输出输入数据的均值
  print('Mean of train-time output: ', out.mean()) # 输出训练时输出的均值
  print('Mean of test-time output: ', out_test.mean()) # 输出测试时输出的均值
  print('Fraction of train-time output set to zero: ', (out == 0).mean())  # 输出训练时被置为零的比例
  print('Fraction of test-time output set to zero: ', (out_test == 0).mean())  # 输出测试时被置为零的比例
  print()


# # Dropout 反向传播
# 在文件 `cs231n/layers.py` 中实现 dropout 的反向传播。完成后运行以下单元格以进行数值梯度检查。
x = np.random.randn(10, 10) + 10  # 随机生成一个 10x10 的矩阵
dout = np.random.randn(*x.shape)  # 随机生成一个与 x 相同形状的矩阵

dropout_param = {'mode': 'train', 'p': 0.8, 'seed': 123}  # 设置 dropout 参数
out, cache = dropout_forward(x, dropout_param)  # 前向传播
dx = dropout_backward(dout, cache)  # 反向传播
dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)  # 数值梯度

print('输出3：测试反向传播（输出梯度误差）')
print('dx relative error: ', rel_error(dx, dx_num))

# # 带有 Dropout 的全连接网络
# 在文件 `cs231n/classifiers/fc_net.py` 中修改实现以使用 dropout。具体而言，
# 如果网络构造函数接收到非零的 dropout 参数，则网络应在每个 ReLU 非线性之后添加 dropout。完成后运行以下内容以进行数值梯度检查。
N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N,))

print('输出4：不同dropout参数下初始损失值和参数的相对误差')
for dropout in [0, 0.25, 0.5]:
  print('Running check with dropout = ', dropout)
  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                            weight_scale=5e-2, dtype=np.float64,
                            dropout=dropout, seed=123) # 创建全连接网络

  loss, grads = model.loss(X, y) # 计算损失和梯度
  print('Initial loss: ', loss)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0] # 定义损失函数
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5) # 数值梯度
    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))) # 输出每个参数的相对误差
  print()

# # 正则化实验
# 作为实验，我们将训练一对两层网络，使用 500 个训练样本：一个不使用 dropout，另一个使用 0.75 的 dropout 概率。
# 然后我们将可视化这两个网络随时间变化的训练和验证准确率。
# 训练两个相同的网络，一个带有 dropout，一个不带 dropout

num_train = 500
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

print('输出5：不同dropout参数下训练过程信息（损失值&准确率）')
solvers = {}  # 存储不同 dropout 设置下的求解器
dropout_choices = [0, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99]  # 不同的 dropout 概率
for dropout in dropout_choices:
  model = FullyConnectedNet([500], dropout=dropout)
  print(dropout)

  solver = Solver(model, small_data,
                  num_epochs=25, batch_size=100,
                  update_rule='adam',
                  optim_config={
                    'learning_rate': 5e-4,
                  },
                  verbose=True, print_every=100)
  solver.train()
  solvers[dropout] = solver

# 绘制两个模型的训练和验证准确率

print('输出6（图像）：可视化两个模型（是否使用dropout）的训练集准确率和验证集准确率')
train_accs = []
val_accs = []
for dropout in dropout_choices:
  solver = solvers[dropout]
  train_accs.append(solver.train_acc_history[-1])
  val_accs.append(solver.val_acc_history[-1])

plt.subplot(3, 1, 1)
for dropout in dropout_choices:
  plt.plot(solvers[dropout].train_acc_history, 'o', label='%.2f dropout' % dropout)
plt.title('Train accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')

plt.subplot(3, 1, 2)
for dropout in dropout_choices:
  plt.plot(solvers[dropout].val_acc_history, 'o', label='%.2f dropout' % dropout)
plt.title('Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')

plt.gcf().set_size_inches(15, 15)
plt.show()


# # Question
# 解释你在这个实验中看到的结果。它对 dropout 有什么启示？
# # Answer
# Dropout 有助于防止过拟合。在不使用 dropout 的情况下，训练集和验证集之间的准确率差距可能达到 65%。如果使用 dropout，
# 随着 p 值的增大，训练集和验证集之间的准确率差距会变小。
# 然而，由于 dropout 会降低神经网络的容量，如果 p 值过大，网络可能会过于弱，无法很好地拟合数据。


