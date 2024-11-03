# # 批量归一化（Batch Normalization）
# 使深层网络更易于训练的一个方法是使用更复杂的优化过程，比如SGD+动量、RMSProp或Adam。
# 另一种策略是改变网络的结构以简化训练。批量归一化（Batch Normalization）是最近提出的一种方法[3]。

# 批量归一化的思想相对简单。通常机器学习方法在输入数据的特征为零均值和单位方差时效果较好。
# 在训练神经网络时，可以在输入数据进入网络之前预处理以显式去相关特征，这样可以保证第一层网络
# 能够接收到服从良好分布的数据。然而，即使我们对输入数据进行预处理，网络深层的激活可能不再
# 是去相关的，并且不再是零均值和单位方差，因为它们受前一层网络输出的影响。
# 更糟糕的是，在训练过程中，由于每层权重的更新，网络中每层特征的分布会发生变化。

# [3]的作者假设网络内部特征分布的变化可能使得深层网络训练变得更加困难。为了解决这个问题，
# [3]建议在网络中插入批量归一化层。在训练时，批量归一化层使用小批量数据估计每个特征的均值和
# 标准差，然后使用这些估计值对小批量特征进行居中和归一化。在训练过程中，使用这些均值和标准差
# 的滑动平均值，而在测试时则使用这些滑动平均值来进行特征的归一化。

# 这种归一化策略可能会减少网络的表达能力，因为某些层的特征可能最佳时不为零均值或单位方差。
# 因此，批归一化层包含了可学习的缩放和偏移参数，来对每个特征维度进行调节。

# [3] Sergey Ioffe 和 Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing
# Internal Covariate Shift", ICML 2015.

import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0) # 设置默认的绘图大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
  """ 返回相对误差 """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# 加载CIFAR10数据（已预处理）

data = get_CIFAR10_data()
print('输出1：训练集、验证集以及测试集形状')
for k, v in data.items():
  print('%s: ' % k, v.shape)


# ## 批量归一化：前向传播
# 在文件`cs231n/layers.py`中，实现批归一化的前向传播函数`batchnorm_forward`。
# 完成后，运行以下代码来测试实现。

# 通过检查批归一化前后特征的均值和方差来验证训练时的前向传播

# 模拟一个两层网络的前向传播
N, D1, D2, D3 = 200, 50, 60, 3  # N是样本数，D1是输入特征维度，D2是第一层输出维度，D3是第二层输出维度
X = np.random.randn(N, D1)  # 生成一个随机输入矩阵X
W1 = np.random.randn(D1, D2)  # 随机初始化第一层权重W1
W2 = np.random.randn(D2, D3)  # 随机初始化第二层权重W2
a = np.maximum(0, X.dot(W1)).dot(W2)  # 计算前向传播，使用ReLU激活函数

print('输出2：批量归一化前后均值标准差对比')
print('Before batch normalization:')
print('  means: ', a.mean(axis=0))
print('  stds: ', a.std(axis=0))

# 均值应接近零，标准差应接近一
print('After batch normalization (gamma=1, beta=0)')
a_norm, _ = batchnorm_forward(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})
print('  mean: ', a_norm.mean(axis=0))
print('  std: ', a_norm.std(axis=0))

# 现在的均值应接近beta，标准差应接近gamma
gamma = np.asarray([1.0, 2.0, 3.0])  # 非平凡的gamma值
beta = np.asarray([11.0, 12.0, 13.0])  # 非平凡的beta值
a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
print('After batch normalization (nontrivial gamma, beta)')
print('  means: ', a_norm.mean(axis=0))
print('  stds: ', a_norm.std(axis=0))

# 检查测试时的前向传播，通过多次运行训练时的前向传播
# 来预热运行平均值，然后在测试时的前向传播之后
# 检查激活的均值和方差。

N, D1, D2, D3 = 200, 50, 60, 3
W1 = np.random.randn(D1, D2)
W2 = np.random.randn(D2, D3)

bn_param = {'mode': 'train'} # 设置批量归一化参数为训练模式
gamma = np.ones(D3)  # 初始化gamma为1
beta = np.zeros(D3)  # 初始化beta为0
for t in range(50): # 重复50次以预热运行平均值
  X = np.random.randn(N, D1) # 生成一个随机输入矩阵X
  a = np.maximum(0, X.dot(W1)).dot(W2) # 计算前向传播，使用ReLU激活函数
  batchnorm_forward(a, gamma, beta, bn_param)

bn_param['mode'] = 'test'  # 切换到测试模式
X = np.random.randn(N, D1)  # 生成新的随机输入矩阵X
a = np.maximum(0, X.dot(W1)).dot(W2)  # 计算前向传播
a_norm, _ = batchnorm_forward(a, gamma, beta, bn_param)  # 进行批量归一化

# 均值应该接近零，标准差应该接近一，但会比训练时的前向传播更嘈杂。
print('输出3：批量归一化后均值标准差（测试模式下）')
print('After batch normalization (test-time):')
print('  means: ', a_norm.mean(axis=0))
print('  stds: ', a_norm.std(axis=0))


# ## 批量归一化: 反向传播
# 现在在函数`batchnorm_backward`中实现批量归一化的反向传播。
#
# 为了推导反向传播，你应该写出批量归一化的计算图，并在每个中间节点上进行反向传播。
# 有些中间值可能有多个输出分支；确保在反向传播中跨这些分支求和梯度。
#
# 完成后，运行以下代码以数值检查你的反向传播。

# 梯度检查批量归一化反向传播
N, D = 4, 5 # 定义样本数N和特征维度D
x = 5 * np.random.randn(N, D) + 12
gamma = np.random.randn(D)
beta = np.random.randn(D)
dout = np.random.randn(N, D)

bn_param = {'mode': 'train'} # 设置批量归一化参数为训练模式
# 定义函数用于计算前向传播输出
fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]
fg = lambda a: batchnorm_forward(x, gamma, beta, bn_param)[0]
fb = lambda b: batchnorm_forward(x, gamma, beta, bn_param)[0]

# 计算数值梯度
dx_num = eval_numerical_gradient_array(fx, x, dout)  # 对x的梯度
da_num = eval_numerical_gradient_array(fg, gamma, dout)  # 对gamma的梯度
db_num = eval_numerical_gradient_array(fb, beta, dout)  # 对beta的梯度

# 前向传播以获取缓存
_, cache = batchnorm_forward(x, gamma, beta, bn_param)
# 调用反向传播计算梯度
dx, dgamma, dbeta = batchnorm_backward(dout, cache)
# 输出数值梯度与计算梯度的误差
print('输出4：测试反向传播是否准确，输出数值梯度与计算梯度的误差')
print('dx error: ', rel_error(dx_num, dx))
print('dgamma error: ', rel_error(da_num, dgamma))
print('dbeta error: ', rel_error(db_num, dbeta))


# ## 批量归一化：替代反向传播
# 在课堂上我们讨论了两种不同的sigmoid反向传播实现。
# 一种策略是写出由简单操作组成的计算图，并通过所有中间值进行反向传播。
# 另一种策略是在纸上推导出导数。
# 对于sigmoid函数，结果表明可以通过简化导数来推导出非常简单的反向传播公式。
#
# 令人惊讶的是，如果在纸上推导并简化导数，批归一化反向传播也可以得出简单表达式。
# 完成后在函数`batchnorm_backward_alt`中实现简化的批量归一化反向传播，并通过运行以下代码比较这两个实现。
# 你的两个实现应该计算出几乎相同的结果，但替代实现应该稍微快一些。
#
# 注意：如果你不能完成这一部分，仍然可以完成作业的其余部分，所以不要太担心。

N, D = 100, 500
x = 5 * np.random.randn(N, D) + 12
gamma = np.random.randn(D)
beta = np.random.randn(D)
dout = np.random.randn(N, D)

bn_param = {'mode': 'train'} # 设置批量归一化参数为训练模式
out, cache = batchnorm_forward(x, gamma, beta, bn_param) # 前向传播获取输出和缓存

t1 = time.time() # 记录时间
dx1, dgamma1, dbeta1 = batchnorm_backward(dout, cache)
t2 = time.time()
dx2, dgamma2, dbeta2 = batchnorm_backward_alt(dout, cache)
t3 = time.time()

# 输出两个实现之间的梯度差异
print('输出5：标准批量归一化反向传播与简化的批量化反向传播的差异')
print('dx difference: ', rel_error(dx1, dx2))
print('dgamma difference: ', rel_error(dgamma1, dgamma2))
print('dbeta difference: ', rel_error(dbeta1, dbeta2))
print('speedup: %.2fx' % ((t2 - t1) / (t3 - t2))) # 输出速度提升倍数


# ## 全连接网络与批量归一化
# 现在你已经有了批归一化的工作实现，返回到 `FullyConnectedNet` 类，在文件 `cs2312n/classifiers/fc_net.py` 中。
# 修改你的实现以添加批量归一化。

# 具体来说，当构造函数中的标志 `use_batchnorm` 为 `True` 时，你应该在每个 ReLU 非线性激活之前插入一个批量归一化层。
# 网络最后一层的输出不应进行归一化。完成后，运行以下代码以检查你的实现的梯度。
# 提示：你可能会发现定义一个额外的辅助层（类似于 `cs231n/layer_utils.py` 文件中的那些）会很有用。
# 如果你决定这样做，请在文件 `cs231n/classifiers/fc_net.py` 中进行。

# 定义输入参数
N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N,))
print('输出6：不同正则化参数下初始损失和梯度检查')
for reg in [0, 3.14]: # 对于不同的正则化参数进行检查
  print('Running check with reg = ', reg)
  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                            reg=reg, weight_scale=5e-2, dtype=np.float64,
                            use_batchnorm=True)

  loss, grads = model.loss(X, y)# 计算损失和梯度
  print('Initial loss: ', loss)  # 打印初始损失

  for name in sorted(grads):  # 遍历所有梯度
      f = lambda _: model.loss(X, y)[0]  # 定义损失函数
      grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)  # 计算数值梯度
      print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))  # 打印相对误差
  if reg == 0: print()  # 当正则化为0时换行

# # 深度网络的批量归一化
# # 运行以下代码，以批量归一化和不使用批量归一化的方式训练一个六层网络，使用1000个训练样本的子集。
# # 尝试训练一个非常深的网络并使用批量归一化
hidden_dims = [100, 100, 100, 100, 100]   # 隐藏层维度

num_train = 1000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

weight_scale = 2e-2  # 权重缩放因子
bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)
model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)

# 初始化带批量归一化的模型训练器
bn_solver = Solver(bn_model, small_data,
                num_epochs=10, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=200)
print('输出7：带批量归一化的模型训练过程')
bn_solver.train()

# 初始化不带批量归一化的模型训练器
solver = Solver(model, small_data,
                num_epochs=10, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=200)
print('输出8：不带批量归一化的模型训练过程')
solver.train()

# 运行以下代码以可视化上面训练的两个网络的结果。你应该会发现，使用批归一化有助于网络更快收敛。
print('输出9（图像）：可视化有无批量归一化处理的训练损失变化、训练集准确率和验证集准确率变化')
plt.subplot(3, 1, 1)
plt.title('Training loss')
plt.xlabel('Iteration')

plt.subplot(3, 1, 2)
plt.title('Training accuracy')
plt.xlabel('Epoch')

plt.subplot(3, 1, 3)
plt.title('Validation accuracy')
plt.xlabel('Epoch')

plt.subplot(3, 1, 1)
plt.plot(solver.loss_history, 'o', label='baseline')
plt.plot(bn_solver.loss_history, 'o', label='batchnorm')

plt.subplot(3, 1, 2)
plt.plot(solver.train_acc_history, '-o', label='baseline')
plt.plot(bn_solver.train_acc_history, '-o', label='batchnorm')

plt.subplot(3, 1, 3)
plt.plot(solver.val_acc_history, '-o', label='baseline')
plt.plot(bn_solver.val_acc_history, '-o', label='batchnorm')
  
for i in [1, 2, 3]:
  plt.subplot(3, 1, i)
  plt.legend(loc='upper center', ncol=4)
plt.gcf().set_size_inches(15, 15)
plt.show()


# # 批量归一化与初始化
# 现在我们将进行一个小实验，研究批归一化和权重初始化之间的相互作用。
#
# 第一个代码块将训练8层网络，使用不同的权重初始化规模，分别对比有无批归一化的效果。
# 第二个代码块将绘制训练准确率、验证集准确率和训练损失与权重初始化规模的关系。
# 尝试训练一个非常深的网络并使用批归一化
hidden_dims = [50, 50, 50, 50, 50, 50, 50]  # 定义7个隐藏层，每层50个单元

num_train = 1000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}
print('输出10：20个权重规模处理过程')
bn_solvers = {}  # 存储使用批量归一化的求解器
solvers = {}  # 存储不使用批量归一化的求解器
weight_scales = np.logspace(-4, 0, num=20) # 生成20个权重初始化规模，范围从10^-4到10^0
for i, weight_scale in enumerate(weight_scales):
  print('Running weight scale %d / %d' % (i + 1, len(weight_scales)))
  bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)
  model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)

  # 创建带批归一化的求解器并训练
  bn_solver = Solver(bn_model, small_data,
                  num_epochs=10, batch_size=50,
                  update_rule='adam',
                  optim_config={
                    'learning_rate': 1e-3, # 学习率
                  },
                  verbose=False, print_every=200)
  bn_solver.train()  # 训练模型
  bn_solvers[weight_scale] = bn_solver

  # 创建不带批归一化的求解器并训练
  solver = Solver(model, small_data,
                  num_epochs=10, batch_size=50,
                  update_rule='adam',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  verbose=False, print_every=200)
  solver.train()  # 训练模型
  solvers[weight_scale] = solver  # 保存结果

# 绘制权重规模实验结果
best_train_accs, bn_best_train_accs = [], []  # 存储最佳训练准确率
best_val_accs, bn_best_val_accs = [], []  # 存储最佳验证准确率
final_train_loss, bn_final_train_loss = [], []  # 存储最终训练损失

# 提取结果
for ws in weight_scales:
    best_train_accs.append(max(solvers[ws].train_acc_history))  # 获取最佳训练准确率
    bn_best_train_accs.append(max(bn_solvers[ws].train_acc_history))  # 获取带批归一化的最佳训练准确率

    best_val_accs.append(max(solvers[ws].val_acc_history))  # 获取最佳验证准确率
    bn_best_val_accs.append(max(bn_solvers[ws].val_acc_history))  # 获取带批归一化的最佳验证准确率

    final_train_loss.append(np.mean(solvers[ws].loss_history[-100:]))  # 获取最后100次训练损失的平均值
    bn_final_train_loss.append(np.mean(bn_solvers[ws].loss_history[-100:]))  # 获取带批归一化的最后100次训练损失的平均值

print('输出11（图像）：可视化最佳验证集准确率、最佳训练集准确率以及最终训练损失和权重初始化规模的关系')
plt.subplot(3, 1, 1)
plt.title('Best val accuracy vs weight initialization scale')
plt.xlabel('Weight initialization scale')
plt.ylabel('Best val accuracy')
plt.semilogx(weight_scales, best_val_accs, '-o', label='baseline')
plt.semilogx(weight_scales, bn_best_val_accs, '-o', label='batchnorm')
plt.legend(ncol=2, loc='lower right')

plt.subplot(3, 1, 2)
plt.title('Best train accuracy vs weight initialization scale')
plt.xlabel('Weight initialization scale')
plt.ylabel('Best training accuracy')
plt.semilogx(weight_scales, best_train_accs, '-o', label='baseline')
plt.semilogx(weight_scales, bn_best_train_accs, '-o', label='batchnorm')
plt.legend()

plt.subplot(3, 1, 3)
plt.title('Final training loss vs weight initialization scale')
plt.xlabel('Weight initialization scale')
plt.ylabel('Final training loss')
plt.semilogx(weight_scales, final_train_loss, '-o', label='baseline')
plt.semilogx(weight_scales, bn_final_train_loss, '-o', label='batchnorm')
plt.legend()

plt.gcf().set_size_inches(10, 15)
plt.show()


# # Question:
# 描述这个实验的结果，并尝试给出实验结果的原因。

# # Answer:
# 结果显示，使用批归一化的模型可以：
# 1. 对糟糕的初始化更加鲁棒（即使在糟糕的初始化下也能获得良好的性能）。
# 2. 更加有效地避免过拟合（在某些权重规模下，没有批归一化的模型在验证集和训练集之间的差距更大）。
# 3. 预防梯度消失和梯度爆炸问题。
# 我认为原因在于批归一化可以防止各层输入值过小或过大，从而在反向传播过程中网络可以获得更好的梯度。