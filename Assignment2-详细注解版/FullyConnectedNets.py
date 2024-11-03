# # 全连接神经网络
# 在之前的作业中，你在 CIFAR-10 数据集上实现了一个简单的两层全连接神经网络。虽然实现简单，
# 但因为损失和梯度是在一个整体函数中计算的，所以不够模块化。对于简单的两层网络可以管理，但随着模型的增大，这种方式会变得不切实际。
# 理想情况下，我们希望采用更模块化的设计来构建网络，以便可以独立实现不同类型的层，然后将它们组合成不同架构的模型。
#
# 在本次练习中，我们将使用更模块化的方法实现全连接网络。对于每一层，我们将实现一个 `forward` 和一个 `backward` 函数。
# `forward` 函数接收输入、权重和其他参数，并返回输出和一个 `cache` 对象，用于存储反向传递所需的数据，示例如下：
#
# ```python
# def layer_forward(x, w):
#   """ 接收输入 x 和权重 w """
#   # 执行一些计算 ...
#   z = # ... 一些中间值
#   # 再执行一些计算 ...
#   out = # 输出结果
#
#   cache = (x, w, z, out) # 计算梯度所需的值
#
#   return out, cache
# ```
#
# 反向传递将接收上游导数和 `cache` 对象，并返回关于输入和权重的梯度，示例如下：
#
# ```python
# def layer_backward(dout, cache):
#   """
#   接收关于输出的损失导数和 cache，
#   计算关于输入的导数。
#   """
#   # 解包 cache 中的值
#   x, w, z, out = cache
#
#   # 使用 cache 中的值计算导数
#   dx = # 关于 x 的损失导数
#   dw = # 关于 w 的损失导数
#
#   return dx, dw
# ```
#
# 实现多个这样的层后，我们将能够轻松地将它们组合起来构建任意深度的分类器。
#
# 除了实现任意深度的全连接网络，我们还将探索不同的优化更新规则，并引入 Dropout 作为正则化工具，
# Batch Normalization 作为更有效地优化深度网络的工具。

# 按照惯例，进行一些设置

import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0) # 设置图表的默认大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 自动重新加载外部模块
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
def rel_error(x, y):
  """ 返回相对误差 """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# 获取 CIFAR10 数据集
data = get_CIFAR10_data()
print('输出1：训练集、验证集以及测试集形状')
for k, v in data.items():
  print('%s: ' % k, v.shape)


# # 仿射层：前向传播
# 打开文件 `cs231n/layers.py` 并实现 `affine_forward` 函数。
#
# 一旦完成，你可以通过运行以下代码来测试你的实现：

# 测试 affine_forward 函数

num_inputs = 2
input_shape = (4, 5, 6)
output_dim = 3

input_size = num_inputs * np.prod(input_shape) # 计算输入的总大小
weight_size = output_dim * np.prod(input_shape) # 计算权重的总大小

# 创建测试输入数据
x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
b = np.linspace(-0.3, 0.1, num=output_dim)

# 进行前向传播
out, _ = affine_forward(x, w, b)
correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                        [ 3.25553199,  3.5141327,   3.77273342]])

# 将输出与我们的输出进行比较 误差应在 1e-9 附近。
print('输出2：前向传播函数测试结果（通过比较输出矩阵与期望输出矩阵得到）')
print('Testing affine_forward function:')
print('difference: ', rel_error(out, correct_out))


# # 仿射层：反向传播
# 现在实现 `affine_backward` 函数，并使用数值梯度检查来测试您的实现。
#
# 测试 affine_backward 函数

x = np.random.randn(10, 2, 3)  # 随机生成输入数据
w = np.random.randn(6, 5)       # 随机生成权重
b = np.random.randn(5)           # 随机生成偏置
dout = np.random.randn(10, 5)    # 随机生成输出的导数

# 计算数值梯度
dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

# 进行前向传播并获取 cache
_, cache = affine_forward(x, w, b)
dx, dw, db = affine_backward(dout, cache)

# 误差应在 1e-10 之内
print('输出3：反向传播函数测试结果（通过比较数值梯度和反向传播计算的梯度得到）')
print('Testing affine_backward function:')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))


# # ReLU层：前向传播
# 实现 `relu_forward` 函数的前向传播，并使用以下内容测试您的实现：

# 测试 relu_forward 函数

x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

out, _ = relu_forward(x)
correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
                        [ 0.,          0.,          0.04545455,  0.13636364,],
                        [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])


# 将您的输出与我们的输出进行比较。误差应在 1e-8 附近
print('输出4：ReLu前向传播函数测试结果（通过比较输出矩阵和期望矩阵得到）')
print('Testing relu_forward function:')
print('difference: ', rel_error(out, correct_out))



# # ReLU层：反向传播
# 现在实现 `relu_backward` 函数的反向传播，并使用数值梯度检查测试您的实现：

x = np.random.randn(10, 10)  # 随机生成输入数据
dout = np.random.randn(*x.shape)  # 随机生成输出的导数

# 计算数值梯度
dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)

# 进行前向传播并获取 cache
_, cache = relu_forward(x)
dx = relu_backward(dout, cache)

# 误差应在 1e-12 之内
print('输出5：ReLu反向传播函数测试结果（通过比较数值梯度和反向传播计算的梯度得到）')
print('Testing relu_backward function:')
print('dx error: ', rel_error(dx_num, dx))


# # “三明治”层
# 神经网络中有一些常见的层模式。例如，仿射层通常后面跟着 ReLU 非线性。
# 为了方便这些常见模式，我们在 `cs231n/layer_utils.py` 中定义了几个方便的层。
#
# 现在查看 `affine_relu_forward` 和 `affine_relu_backward` 函数，并运行以下内容来对反向传播进行数值梯度检查：

from cs231n.layer_utils import affine_relu_forward, affine_relu_backward


x = np.random.randn(2, 3, 4)  # 随机生成输入数据
w = np.random.randn(12, 10)    # 随机生成权重
b = np.random.randn(10)         # 随机生成偏置
dout = np.random.randn(2, 10)   # 随机生成输出的导数

# 进行前向传播并获取 cache
out, cache = affine_relu_forward(x, w, b)
# 进行反向传播
dx, dw, db = affine_relu_backward(dout, cache)
# 计算数值梯度
dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)

# 误差应在 1e-10 之内
print('输出6：测试affine_relu_forward 和 affine_relu_backward 函数（通过比较数值梯度和反向传播得到的梯度得到）')
print('Testing affine_relu_forward:')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

# #损失层：Softmax 和 SVM
# 你在上一个作业中实现了这些损失函数，因此我们在这里提供给你。你仍然应该通过查看 cs231n/layers.py 中的实现来确保你理解它们是如何工作的。
#
# 你可以通过运行以下代码来确保实现是正确的：

num_classes, num_inputs = 10, 50
x = 0.001 * np.random.randn(num_inputs, num_classes)
y = np.random.randint(num_classes, size=num_inputs)

# 计算 SVM 损失的数值梯度
dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
loss, dx = svm_loss(x, y)

# 测试 svm_loss 函数。损失应接近 9，dx 误差应为 1e-9
print('输出7：测试SVM损失函数的正确性（通过输出损失和梯度误差得到）')
print('Testing svm_loss:')
print('loss: ', loss)
print('dx error: ', rel_error(dx_num, dx))

# 计算 Softmax 损失的数值梯度
dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
loss, dx = softmax_loss(x, y)

# 测试 softmax_loss 函数。损失应为 2.3，dx 误差应为 1e-8
print('输出8：测试softmax损失函数的正确性（通过输出损失和梯度误差得到）')
print('\nTesting softmax_loss:')
print('loss: ', loss)
print('dx error: ', rel_error(dx_num, dx))


# # 两层网络
# 在上一个作业中，你以单一的整体类实现了一个两层神经网络。现在你将使用这些模块化实现重新实现两层网络。
#
# 打开文件 cs231n/classifiers/fc_net.py 并完成 TwoLayerNet 类的实现。这个类将作为你在本次作业中实现的其他网络的模型，
# 因此请仔细阅读以确保你理解 API。你可以运行下面的单元来测试你的实现。

N, D, H, C = 3, 5, 50, 7
X = np.random.randn(N, D)
y = np.random.randint(C, size=N)

std = 1e-2
model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)
print('输出9：测试两层网络模型的正确性（初始化、前向传播、损失计算、反向传播的梯度检查）')
print('Testing initialization ... ')
W1_std = abs(model.params['W1'].std() - std)
b1 = model.params['b1']
W2_std = abs(model.params['W2'].std() - std)
b2 = model.params['b2']
# 检查第一层的权重和偏置
assert W1_std < std / 10, 'First layer weights do not seem right'
assert np.all(b1 == 0), 'First layer biases do not seem right'
# 检查第二层的权重和偏置
assert W2_std < std / 10, 'Second layer weights do not seem right'
assert np.all(b2 == 0), 'Second layer biases do not seem right'

print('Testing test-time forward pass ... ')
# 设置模型的权重和偏置
model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
# 测试数据的线性空间
X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
scores = model.loss(X) # 计算得分
correct_scores = np.asarray(
  [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
   [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
   [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
scores_diff = np.abs(scores - correct_scores).sum()
# 检查前向传播的正确性
assert scores_diff < 1e-6, 'Problem with test-time forward pass'

print('Testing training loss (no regularization)')
y = np.asarray([0, 5, 1])
loss, grads = model.loss(X, y)
correct_loss = 3.4702243556
# 检查损失值
assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'

# 测试带正则化的损失
model.reg = 1.0
loss, grads = model.loss(X, y)
correct_loss = 26.5948426952
assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'

for reg in [0.0, 0.7]:
  print('Running numeric gradient check with reg = ', reg)
  model.reg = reg
  loss, grads = model.loss(X, y)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))

# # 求解器
# 在上一个作业中，训练模型的逻辑与模型本身耦合在一起。遵循更模块化的设计，对于本次作业，我们将训练模型的逻辑拆分到一个单独的类中。
#
# 打开文件 `cs231n/solver.py`，并阅读以熟悉 API。完成后，
# 使用 `Solver` 实例训练一个 `TwoLayerNet`，使其在验证集上达到至少 `50%` 的准确率。

model = TwoLayerNet(reg=1e-1)
solver = None
print('输出10：训练 TwoLayerNet 模型的过程中每个训练周期（epoch）和迭代（iteration）的训练准确率、验证准确率以及损失值的变化情况')
##############################################################################
# TODO:使用 Solver 实例训练一个 TwoLayerNet，使其在验证集上达到至少50%的准确率  #
##############################################################################
solver = Solver(model, data,
    update_rule='sgd',
    optim_config={
    'learning_rate': 1e-3,
    },
    lr_decay=0.8,
    num_epochs=10, batch_size=100,
    print_every=100)
solver.train()
scores = model.loss(data['X_test'])
y_pred = np.argmax(scores, axis = 1)
acc = np.mean(y_pred == data['y_test'])
print('输出11：模型在测试集上的准确率')
print('test acc: %f' %(acc))

# 运行此单元以可视化训练损失和训练/验证准确率
print('输出12（图像）：训练损失&训练集、验证集的准确率可视化')
plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(solver.loss_history, 'o')
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()

# # 多层网络
# 接下来你将实现一个具有任意数量隐藏层的全连接网络。
#
# 阅读文件 `cs231n/classifiers/fc_net.py` 中的 `FullyConnectedNet` 类。
#
# 实现初始化、前向传播和反向传播。暂时不必担心实现 dropout 或批量归一化；我们将很快添加这些功能。

# ## 初始损失和梯度检查

# 作为一个完整性检查，运行以下代码以检查初始损失，并检查网络在有无正则化情况下的梯度。初始损失看起来合理吗？
#
# 对于梯度检查，你应该期望误差在 1e-6 或更小。


N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N,))
print('输出13：对全连接神经网络模型的损失和梯度进行检查')
for reg in [0, 3.14]:
  print('Running check with reg = ', reg) # 正则化强度
  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                            reg=reg, weight_scale=5e-2, dtype=np.float64)

  loss, grads = model.loss(X, y)
  print('Initial loss: ', loss)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))



# 作为另一个完整性检查，确保你可以在一个小的数据集上进行过拟合，该数据集包含 50 张图像。
# 首先我们将尝试一个具有 100 个单元的三层网络。你需要调整学习率和初始化规模，但应该能够在 20 轮内实现 100% 的训练准确率。
print('使用小数据集（50张图像）进行完整性检查')
print('三层网络效果：')
# TODO: 使用三层网络对 50 个训练样本进行过拟合。
print('输出14：训练过程中每个周期的训练集准确率和验证集准确率，以及某些迭代点的损失值（10个1输出）')
num_train = 50
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

weight_scale = 1e-2
learning_rate = 8e-3
model = FullyConnectedNet([100, 100],
              weight_scale=weight_scale, dtype=np.float64)
solver = Solver(model, small_data,
                print_every=10, num_epochs=20, batch_size=25,
                update_rule='sgd',
                optim_config={
                  'learning_rate': learning_rate,
                }
         )
solver.train()
print('输出15（图像）：训练过程中每次迭代的损失值变化')
plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.show()


# 现在尝试使用一个五层网络，每层有 100 个单元来对 50 个训练样本进行过拟合。再次调整学习率和权重初始化，
# 但应该能够在 20 个 epoch 内实现 100% 的训练准确率。

# TODO: 使用五层网络对 50 个训练样本进行过拟合。
print('五层网络效果：')
num_train = 50
print('输出16：训练过程中每个周期的训练集准确率和验证集准确率，以及某些迭代点的损失值（10个1输出）')
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

learning_rate = 3e-4
weight_scale = 1e-1 # 权重初始化的尺度
model = FullyConnectedNet([100, 100, 100, 100], # 创建一个五层全连接网络
                weight_scale=weight_scale, dtype=np.float64)
solver = Solver(model, small_data,
                print_every=10, num_epochs=20, batch_size=25,
                update_rule='sgd',
                optim_config={
                  'learning_rate': learning_rate,
                }
         )
solver.train()
print('输出17（图像）：训练过程中每次迭代的损失值变化')
plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.show()


# # 相关问题：
# 你注意到三层网络与五层网络的训练难度比较有什么不同吗？
# 
# # 答案：
#
# 在调整五层网络的超参数时，我发现最小损失对权重尺度更加敏感，这使得损失函数更容易收敛到局部最小值。我认为这个现象发生是因为
# 五层网络的高容量使得其损失函数比三层网络更加复杂，并且更难找到最优域，而且对初始化的敏感性更高。

# # 更新规则
# 到目前为止，我们使用的是普通的随机梯度下降（SGD）作为更新规则。更复杂的更新规则可以使训练深层网络更容易。我们将实现一些最常用的更新规则，
# 并将它们与普通 SGD 进行比较。

# # SGD+动量
# 带动量的随机梯度下降是一种广泛使用的更新规则，通常比普通的随机梯度下降更快收敛。
#
# 打开文件 `cs231n/optim.py`，并阅读文件顶部的文档，以确保理解 API。在 `sgd_momentum` 函数中实现 SGD+动量更新规则，
# 然后运行以下代码以检查你的实现。你应该看到误差小于 1e-8。

from cs231n.optim import sgd_momentum

N, D = 4, 5
w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
v = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)

config = {'learning_rate': 1e-3, 'velocity': v}
next_w, _ = sgd_momentum(w, dw, config=config)

expected_next_w = np.asarray([
  [ 0.1406,      0.20738947,  0.27417895,  0.34096842,  0.40775789],
  [ 0.47454737,  0.54133684,  0.60812632,  0.67491579,  0.74170526],
  [ 0.80849474,  0.87528421,  0.94207368,  1.00886316,  1.07565263],
  [ 1.14244211,  1.20923158,  1.27602105,  1.34281053,  1.4096    ]])
expected_velocity = np.asarray([
  [ 0.5406,      0.55475789,  0.56891579, 0.58307368,  0.59723158],
  [ 0.61138947,  0.62554737,  0.63970526,  0.65386316,  0.66802105],
  [ 0.68217895,  0.69633684,  0.71049474,  0.72465263,  0.73881053],
  [ 0.75296842,  0.76712632,  0.78128421,  0.79544211,  0.8096    ]])

print('输出18：权重更新结果与预期结果的相对误差&更新速度和预期速度的相对误差（验证SGD+动量优化器的实现）')
print('next_w error: ', rel_error(next_w, expected_next_w))
print('velocity error: ', rel_error(expected_velocity, config['velocity']))



# 一旦你完成了，就运行以下代码来训练一个六层网络，分别使用 SGD 和 SGD+动量。你应该会看到 SGD+动量更新规则收敛更快。
num_train = 4000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

solvers = {}
print('输出19：SGD和带动量的SGD两种优化器的训练过程')
for update_rule in ['sgd', 'sgd_momentum']:
  print('running with ', update_rule)
  model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)

  solver = Solver(model, small_data,
                  num_epochs=5, batch_size=100,
                  update_rule=update_rule,
                  optim_config={
                    'learning_rate': 1e-2,
                  },
                  verbose=True)
  solvers[update_rule] = solver
  solver.train()
  print()
print('输出20（图像）：两种优化器训练损失随迭代次数的变化、训练集准确率的变化、验证集准确率的变化')
plt.subplot(3, 1, 1)
plt.title('Training loss')
plt.xlabel('Iteration')

plt.subplot(3, 1, 2)
plt.title('Training accuracy')
plt.xlabel('Epoch')

plt.subplot(3, 1, 3)
plt.title('Validation accuracy')
plt.xlabel('Epoch')

for update_rule, solver in solvers.items():
  plt.subplot(3, 1, 1)
  plt.plot(solver.loss_history, 'o', label=update_rule)
  
  plt.subplot(3, 1, 2)
  plt.plot(solver.train_acc_history, '-o', label=update_rule)

  plt.subplot(3, 1, 3)
  plt.plot(solver.val_acc_history, '-o', label=update_rule)
  
for i in [1, 2, 3]:
  plt.subplot(3, 1, i)
  plt.legend(loc='upper center', ncol=4)
plt.gcf().set_size_inches(15, 15)
plt.show()


# # RMSProp 和 Adam
# RMSProp 和 Adam 是更新规则，通过使用梯度的二次矩阵的运行平均来设置每个参数的学习率。

# 在文件 `cs231n/optim.py` 中，实现 `rmsprop` 函数中的 RMSProp 更新规则，以及在 `adam` 函数中实现 Adam 更新规则，并使用以下测试检查你的实现。

# [1] Tijmen Tieleman 和 Geoffrey Hinton. "Lecture 6.5-rmsprop: 将梯度除以其近期大小的运行平均." COURSERA: Neural Networks for Machine Learning 4 (2012)。

# [2] Diederik Kingma 和 Jimmy Ba, "Adam: 一种随机优化方法", ICLR 2015。

# 测试 RMSProp 实现；你应该看到误差小于 1e-7
from cs231n.optim import rmsprop

N, D = 4, 5
# 创建权重 w，范围从 -0.4 到 0.6，并将其重塑为 (4, 5) 的形状
w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
# 创建梯度 dw，范围从 -0.6 到 0.4，并将其重塑为 (4, 5) 的形状
dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
# 创建缓存 cache，范围从 0.6 到 0.9，并将其重塑为 (4, 5) 的形状
cache = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)

# 配置字典，包含学习率和缓存
config = {'learning_rate': 1e-2, 'cache': cache}
# 使用 RMSProp 更新权重

next_w, _ = rmsprop(w, dw, config=config)
# 期望的下一个权重
expected_next_w = np.asarray([
  [-0.39223849, -0.34037513, -0.28849239, -0.23659121, -0.18467247],
  [-0.132737,   -0.08078555, -0.02881884,  0.02316247,  0.07515774],
  [ 0.12716641,  0.17918792,  0.23122175,  0.28326742,  0.33532447],
  [ 0.38739248,  0.43947102,  0.49155973,  0.54365823,  0.59576619]])
# 期望的缓存
expected_cache = np.asarray([
  [ 0.5976,      0.6126277,   0.6277108,   0.64284931,  0.65804321],
  [ 0.67329252,  0.68859723,  0.70395734,  0.71937285,  0.73484377],
  [ 0.75037008,  0.7659518,   0.78158892,  0.79728144,  0.81302936],
  [ 0.82883269,  0.84469141,  0.86060554,  0.87657507,  0.8926    ]])

print('输出21：测试RMSProp优化器的实现（更新后权重与预期权重对比&更新后缓存与期望缓存的对比）')
# 检查 next_w 的误差
print('next_w error: ', rel_error(expected_next_w, next_w))
# 检查缓存的误差
print('cache error: ', rel_error(expected_cache, config['cache']))

# 测试 Adam 实现；你应该看到误差在 1e-7 或更小的范围内
from cs231n.optim import adam

N, D = 4, 5
# 创建权重 w，范围从 -0.4 到 0.6，并将其重塑为 (4, 5) 的形状
w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
# 创建梯度 dw，范围从 -0.6 到 0.4，并将其重塑为 (4, 5) 的形状
dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)

# 创建一阶矩 m，范围从 0.6 到 0.9，并将其重塑为 (4, 5) 的形状
m = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)
# 创建二阶矩 v，范围从 0.7 到 0.5，并将其重塑为 (4, 5) 的形状
v = np.linspace(0.7, 0.5, num=N*D).reshape(N, D)

# 配置字典，包含学习率，m 和 v
config = {'learning_rate': 1e-2, 'm': m, 'v': v, 't': 5}
# 使用 Adam 更新权重
next_w, _ = adam(w, dw, config=config)

# 期望的下一个权重
expected_next_w = np.asarray([
  [-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
  [-0.1380274,  -0.08544591, -0.03286534,  0.01971428,  0.0722929],
  [ 0.1248705,   0.17744702,  0.23002243,  0.28259667,  0.33516969],
  [ 0.38774145,  0.44031188,  0.49288093,  0.54544852,  0.59801459]])
# 期望的二阶矩 v
expected_v = np.asarray([
  [ 0.69966,     0.68908382,  0.67851319,  0.66794809,  0.65738853,],
  [ 0.64683452,  0.63628604,  0.6257431,   0.61520571,  0.60467385,],
  [ 0.59414753,  0.58362676,  0.57311152,  0.56260183,  0.55209767,],
  [ 0.54159906,  0.53110598,  0.52061845,  0.51013645,  0.49966,   ]])
# 期望的一阶矩 m
expected_m = np.asarray([
  [ 0.48,        0.49947368,  0.51894737,  0.53842105,  0.55789474],
  [ 0.57736842,  0.59684211,  0.61631579,  0.63578947,  0.65526316],
  [ 0.67473684,  0.69421053,  0.71368421,  0.73315789,  0.75263158],
  [ 0.77210526,  0.79157895,  0.81105263,  0.83052632,  0.85      ]])

# 检查误差
print('输出22：测试Adam优化器的实现：更新后权重，梯度均值和梯度方差与预期值之间的误差')
print('next_w error: ', rel_error(expected_next_w, next_w))
print('v error: ', rel_error(expected_v, config['v']))
print('m error: ', rel_error(expected_m, config['m']))


# 一旦你调试好了 RMSProp 和 Adam 的实现，运行以下代码来使用这些新的更新规则训练一对深度网络：

learning_rates = {'rmsprop': 1e-4, 'adam': 1e-3}
# 遍历更新规则
print('输出23：RMSProp和Adam两种优化器下分别训练全连接神经网络（训练过程）')
for update_rule in ['adam', 'rmsprop']:
  print('running with ', update_rule)
  # 初始化一个五层的全连接网络
  model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)

  # 创建求解器
  solver = Solver(model, small_data,
                  num_epochs=5, batch_size=100,
                  update_rule=update_rule,
                  optim_config={
                    'learning_rate': learning_rates[update_rule]
                  },
                  verbose=True)
  # 训练模型
  solvers[update_rule] = solver
  solver.train()
  print()

# 绘制训练损失图、训练准确率和验证准确率的图表
plt.subplot(3, 1, 1)
plt.title('Training loss')
plt.xlabel('Iteration')

plt.subplot(3, 1, 2)
plt.title('Training accuracy')
plt.xlabel('Epoch')

plt.subplot(3, 1, 3)
plt.title('Validation accuracy')
plt.xlabel('Epoch')
print('输出24（图像）：可视化每种优化算法下的训练损失、训练集准确率以及验证集准确率')
# 绘制每个更新规则下的损失和准确率 三个子图分别显示训练损失，训练准确率，验证准确率
for update_rule, solver in solvers.items():
  plt.subplot(3, 1, 1)
  plt.plot(solver.loss_history, 'o', label=update_rule)
  
  plt.subplot(3, 1, 2)
  plt.plot(solver.train_acc_history, '-o', label=update_rule)

  plt.subplot(3, 1, 3)
  plt.plot(solver.val_acc_history, '-o', label=update_rule)
  
for i in [1, 2, 3]:
  plt.subplot(3, 1, i)
  plt.legend(loc='upper center', ncol=4)
plt.gcf().set_size_inches(15, 15)
plt.show()


# # 训练一个好的模型！
# 训练你能在 CIFAR-10 上训练的最佳全连接模型，并将你的最佳模型存储在 `best_model` 变量中。
# 我们要求你在验证集上获得至少 50% 的准确率。
#
# 如果你仔细的话，应该可以达到 55% 以上的准确率，但这部分不要求这样做，也不会给予额外的学分。
# 在这个作业的后面部分，我们将要求你在 CIFAR-10 上训练你能得到的最佳卷积网络，我们希望你将精力放在卷积网络上，而不是全连接网络上。
#
# 在完成这部分之前，可能会发现完成 `BatchNormalization.ipynb` 和 `Dropout.ipynb` 笔记本非常有用，因为这些技术可以帮助你训练强大的模型。
print('输出25：在数据集CIFAR-10上的训练过程')
best_model = None  # 初始化最佳模型为 None
################################################################################
# TODO: 在 CIFAR-10 上训练你能得到的最佳 FullyConnectedNet。你可能会发现批量归一化和 dropout 非常有用。
#  将你的最佳模型存储在best_model 变量中。                                                         #
################################################################################
X_val= data['X_val']
y_val= data['y_val']
X_test= data['X_test']
y_test= data['y_test']

learning_rate = 3.1e-4 # 设置学习率
weight_scale = 2.5e-2 # 设置权重缩放参数
# 创建一个五层的全连接网络
model = FullyConnectedNet([600, 500, 400, 300, 200, 100],
                weight_scale=weight_scale, dtype=np.float64, dropout=0.25, use_batchnorm=True, reg=1e-2)
# 创建求解器，配置训练参数
solver = Solver(model, data,
                print_every=500, num_epochs=30, batch_size=100,
                update_rule='adam',
                optim_config={
                  'learning_rate': learning_rate,
                },
                lr_decay=0.9
         )

solver.train()  # 训练模型
scores = model.loss(data['X_test'])  # 计算测试集上的损失
y_pred = np.argmax(scores, axis = 1)  # 获取预测的标签
acc = np.mean(y_pred == data['y_test'])  # 计算测试集的准确率
print('输出26：在测试集上的准确率')
print('test acc: %f' %(acc))  # 输出测试集准确率
best_model = model  # 将当前模型设为最佳模型
# 设定参数网格
# learning_rates = [1e-5, 1e-4, 3.1e-4, 1e-3]
# weight_scales = [1e-2, 2.5e-2, 5e-2, 1e-1]
# regs = [0, 1e-2, 1e-1, 1]
# batch_sizes = [32, 64, 128, 256]
# dropout_rates = [0.1, 0.25, 0.5]

# 记录每种组合的测试集准确率
# results = []
# best_model = None  # 初始化 best_model 为 None
# best_acc = 0  # 初始化最佳准确率
#
# for lr in learning_rates:
#     for ws in weight_scales:
#         for reg in regs:
#             for bs in batch_sizes:
#                 for dr in dropout_rates:
#                     # 创建模型
#                     model = FullyConnectedNet([600, 500, 400, 300, 200, 100],
#                                               weight_scale=ws, dtype=np.float64,
#                                               dropout=dr, use_batchnorm=True, reg=reg)
#                     # 创建求解器
#                     solver = Solver(model, data,
#                                     print_every=500, num_epochs=10, batch_size=bs,
#                                     update_rule='adam',
#                                     optim_config={'learning_rate': lr},
#                                     lr_decay=0.9)
#
#                     # 训练模型
#                     solver.train()
#
#                     # 计算测试集准确率
#                     scores = model.loss(data['X_test'])
#                     y_pred = np.argmax(scores, axis=1)
#                     acc = np.mean(y_pred == data['y_test'])
#
#                     # 保存参数组合和对应准确率
#                     results.append({
#                         'learning_rate': lr,
#                         'weight_scale': ws,
#                         'reg': reg,
#                         'batch_size': bs,
#                         'dropout_rate': dr,
#                         'test_acc': acc
#                     })
#
#                     if acc > best_acc:
#                         best_acc = acc
#                         best_model = model

# 可视化结果
# learning_rates = [str(lr) for lr in learning_rates]
# weight_scales = [str(ws) for ws in weight_scales]
# regs = [str(r) for r in regs]
# batch_sizes = [str(bs) for bs in batch_sizes]
# dropout_rates = [str(dr) for dr in dropout_rates]

# # 创建图表展示不同参数组合下的测试集准确率
# fig, ax = plt.subplots(figsize=(12, 8))
#
# # 按参数组合准确率排序并绘制
# results_sorted = sorted(results, key=lambda x: x['test_acc'], reverse=True)
# test_accuracies = [res['test_acc'] for res in results_sorted[:10]]  # 展示准确率最高的前10个组合
# param_combinations = [
#     f"LR:{res['learning_rate']}, WS:{res['weight_scale']}, Reg:{res['reg']}, BS:{res['batch_size']}, DR:{res['dropout_rate']}"
#     for res in results_sorted[:10]]
#
# ax.barh(param_combinations, test_accuracies, color='skyblue')
# ax.set_xlabel('Test Accuracy')
# ax.set_title('Top 10 Parameter Combinations by Test Accuracy')
# plt.gca().invert_yaxis()
# plt.show()
#
# # 分析结果
# print("Best Test Accuracy: {:.2f}".format(best_acc))
# print("Best Model Hyperparameters:")
# print("Learning Rate: {}, Weight Scale: {}, Regularization: {}, Batch Size: {}, Dropout Rate: {}".format(
#     best_model.optim_config['learning_rate'],
#     best_model.params['W1'].std(),  # 获取权重标准差，近似表示权重缩放
#     best_model.reg,
#     solver.batch_size,
#     best_model.dropout
# ))

print('输出27（图像）：可视化损失历史、训练集准确率变化以及验证集准确率变化')
# 绘制损失历史
plt.subplot(2, 1, 1)
plt.plot(solver.loss_history)
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

# 绘制分类准确率历史
plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, label='train')
plt.plot(solver.val_acc_history, label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()

# # 测试你的模型
# 在验证集和测试集上运行你的最佳模型。你应该在验证集上达到超过 50% 的准确率。
print('输出28：最佳模型在验证集和测试集上得到的准确率')
y_test_pred = np.argmax(best_model.loss(X_test), axis=1)
y_val_pred = np.argmax(best_model.loss(X_val), axis=1)
print('Validation set accuracy: ', (y_val_pred == y_val).mean())
print('Test set accuracy: ', (y_test_pred == y_test).mean())

