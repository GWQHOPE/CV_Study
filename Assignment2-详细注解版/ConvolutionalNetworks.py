# # 卷积网络
# 到目前为止，我们已经使用深度全连接网络来探索不同的优化策略和网络架构。
# 全连接网络是实验的良好测试平台，因为它们计算效率高，但在实践中所有的最先进结果都使用卷积网络。

# 首先，你将实现几种在卷积网络中使用的层类型。然后，你将使用这些层在 CIFAR-10 数据集上训练一个卷积网络。

# 进行一些初始化设置

import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
# 加载（预处理过的）CIFAR10 数据

print('输出1：训练集、验证集以及测试集形状')
data = get_CIFAR10_data()
for k, v in data.items():
  print('%s: ' % k, v.shape)

# # 卷积：朴素前向传播
# 卷积网络的核心是卷积操作。在文件 `cs231n/layers.py` 中，实现卷积层的前向传播函数 `conv_forward_naive`。
#
# 目前不需要太担心效率；只需以你认为最清晰的方式编写代码。
#
# 你可以通过运行以下代码来测试你的实现：
x_shape = (2, 3, 4, 4)  # 输入形状，包含2个样本，每个样本有3个通道（颜色通道），每个通道4x4像素
w_shape = (3, 3, 4, 4)  # 卷积核形状，包含3个3x3的卷积核，接收4个通道的输入
x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)  # 输入数据
w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)  # 权重（卷积核）
b = np.linspace(-0.1, 0.2, num=3)  # 偏置项

conv_param = {'stride': 2, 'pad': 1}  # 卷积参数：步幅为2，填充为1
out, _ = conv_forward_naive(x, w, b, conv_param) # 执行卷积操作
correct_out = np.array([[[[[-0.08759809, -0.10987781],
                           [-0.18387192, -0.2109216 ]],
                          [[ 0.21027089,  0.21661097],
                           [ 0.22847626,  0.23004637]],
                          [[ 0.50813986,  0.54309974],
                           [ 0.64082444,  0.67101435]]],
                         [[[-0.98053589, -1.03143541],
                           [-1.19128892, -1.24695841]],
                          [[ 0.69108355,  0.66880383],
                           [ 0.59480972,  0.56776003]],
                          [[ 2.36270298,  2.36904306],
                           [ 2.38090835,  2.38247847]]]]])

# 比较你的输出与我们预期的输出；它们之间的差异应该在 1e-8 左右
print('输出2：测试卷积层的前向传播函数（输出卷积层的输出与预期输出的差距）')
print('Testing conv_forward_naive')
print('difference: ', rel_error(out, correct_out))


# # 附录：通过卷积进行图像处理
# 为了检查你的实现并更好地理解卷积层可以执行的操作，我们将设置一个包含两幅图像的输入，
# 并手动设置执行常见图像处理操作的过滤器（灰度转换和边缘检测）。
# 卷积前向传播将这些操作应用于每个输入图像。然后，我们可以将结果可视化，以进行正确性检查。
# from scipy.misc import imread, imresize
import imageio.v2 as imageio
from skimage.transform import resize

# kitten, puppy = imread('kitten.jpg'), imread('puppy.jpg')
kitten, puppy = imageio.imread('kitten.jpg'), imageio.imread('puppy.jpg')
# kitten is wide, and puppy is already square
print('输出3：小猫和小狗图像形状原始大小和裁剪后的大小')
print(kitten.shape , puppy.shape)
d = int(kitten.shape[1] - kitten.shape[0])  # 计算小猫图像的宽度和高度差
# kitten_cropped = kitten[:, d/2:-d/2, :]
kitten_cropped = kitten[:, d//2:-d//2, :]  # 裁剪小猫图像使其与小狗图像相同大小
print(kitten_cropped.shape , puppy.shape)

img_size = 200   # 如果运行太慢，可以将其设置得更小
x = np.zeros((2, 3, img_size, img_size))  # 初始化输入数组
# x[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1))
# x[1, :, :, :] = imresize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))
x[0, :, :, :] = resize(puppy, (img_size, img_size)).transpose((2, 0, 1))
x[1, :, :, :] = resize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))
# 设置一个卷积权重，包含两个3x3的过滤器
w = np.zeros((2, 3, 3, 3))

# 第一个过滤器将图像转换为灰度。
# 设置过滤器的红色、绿色和蓝色通道。
w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]] # 红色通道权重
w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]] # 绿色通道权重
w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]] # 蓝色通道权重

# 第二个过滤器检测蓝色通道中的水平边缘。
w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]  # 边缘检测权重

# 偏置向量。我们不需要灰度过滤器的偏置，
# 但为了边缘检测过滤器，我们希望在每个输出中加128，以避免出现负值。
b = np.array([0, 128])

# 计算卷积操作的结果，将每个输入与每个过滤器相卷积，并加上偏置，将结果存储在 out 中。
out, _ = conv_forward_naive(x, w, b, {'stride': 1, 'pad': 1})

def imshow_noax(img, normalize=True):
    """小帮助函数，用于显示图像并移除坐标轴标签 """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)  # 归一化处理
    plt.imshow(img.astype('uint8')) # 显示图像
    plt.gca().axis('off') # 移除坐标轴

print('输出4（图像）：可视化卷积处理前后的图像（原始图、灰度图和边缘检测图）')
# 显示原始图像和卷积操作的结果
plt.subplot(2, 3, 1)
imshow_noax(puppy, normalize=False)
plt.title('Original image')
plt.subplot(2, 3, 2)
imshow_noax(out[0, 0])
plt.title('Grayscale')
plt.subplot(2, 3, 3)
imshow_noax(out[0, 1])
plt.title('Edges')
plt.subplot(2, 3, 4)
imshow_noax(kitten_cropped, normalize=False)
plt.subplot(2, 3, 5)
imshow_noax(out[1, 0])
plt.subplot(2, 3, 6)
imshow_noax(out[1, 1])
plt.show()


# # 卷积：朴素的反向传播
# 在文件 `cs231n/layers.py` 中实现卷积操作的反向传播函数 `conv_backward_naive`。再次强调，您不需要过于关注计算效率。

# 生成随机输入数据 x，形状为 (4, 3, 5, 5)，表示 4 个样本，3 个通道，每个通道 5x5 的图像
x = np.random.randn(4, 3, 5, 5)
# 生成随机卷积核 w，形状为 (2, 3, 3, 3)，表示 2 个输出通道，3 个输入通道，每个卷积核 3x3
w = np.random.randn(2, 3, 3, 3)
# 生成随机偏置 b，形状为 (2,)，对应于每个输出通道一个偏置
b = np.random.randn(2,)
# 生成随机的梯度 dout，形状为 (4, 2, 5, 5)，表示每个输出通道的梯度
dout = np.random.randn(4, 2, 5, 5)
# 设置卷积参数，包括步幅 stride 和填充 pad
conv_param = {'stride': 1, 'pad': 1}

# 计算输入 x 的数值梯度 dx_num 卷积核 w 的数值梯度 dw_num 偏置 b 的数值梯度 db_num
dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)

# 进行正向传播，计算输出 out 和缓存 cache
out, cache = conv_forward_naive(x, w, b, conv_param)
# 进行反向传播，计算输入的梯度 dx、卷积核的梯度 dw 和偏置的梯度 db
dx, dw, db = conv_backward_naive(dout, cache)

# 误差应该在 1e-9 附近
print('输出5：利用梯度检查测试卷积层的反向传播的实现的准确性')
print('Testing conv_backward_naive function')
print('dx error: ', rel_error(dx, dx_num))
print('dw error: ', rel_error(dw, dw_num))
print('db error: ', rel_error(db, db_num))


# # 最大池化：朴素前向传播
# 在文件 `cs231n/layers.py` 中实现最大池化操作的前向传播函数 `max_pool_forward_naive`。再次强调，您不需要过于关注计算效率。

# 设置输入张量的形状为 (2, 3, 4, 4)，表示 2 个样本，3 个通道，每个通道 4x4 的特征图
x_shape = (2, 3, 4, 4)
# 生成一个从 -0.3 到 0.4 的线性空间数组，并将其重塑为 x_shape 的形状
x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
# 设置池化参数，包括池化的宽度、池化的高度和步幅
pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}
# 进行最大池化的前向传播，得到输出 out 和未使用的缓存 _
out, _ = max_pool_forward_naive(x, pool_param)


correct_out = np.array([[[[-0.26315789, -0.24842105],
                          [-0.20421053, -0.18947368]],
                         [[-0.14526316, -0.13052632],
                          [-0.08631579, -0.07157895]],
                         [[-0.02736842, -0.01263158],
                          [ 0.03157895,  0.04631579]]],
                        [[[ 0.09052632,  0.10526316],
                          [ 0.14947368,  0.16421053]],
                         [[ 0.20842105,  0.22315789],
                          [ 0.26736842,  0.28210526]],
                         [[ 0.32631579,  0.34105263],
                          [ 0.38526316,  0.4       ]]]])

# 将输出与我们的标准输出进行比较。差异应该在 1e-8 左右。
print('输出6：测试最大池化的前向传播函数（实际输出与标准输出的误差）')
print('Testing max_pool_forward_naive function:')
print('difference: ', rel_error(out, correct_out))


# # 最大池化：朴素反向传播
# 在文件 `cs231n/layers.py` 中实现最大池化操作的反向传播函数 `max_pool_backward_naive`。
# 您不需要过于关注计算效率。

# 随机生成一个输入张量 x，形状为 (3, 2, 8, 8)，表示 3 个样本，2 个通道，每个通道 8x8 的特征图
x = np.random.randn(3, 2, 8, 8)
# 随机生成一个输出梯度 dout，形状为 (3, 2, 4, 4)，与池化后的输出对应
dout = np.random.randn(3, 2, 4, 4)
# 设置池化参数，包括池化的高度、宽度和步幅
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

# 使用数值梯度检查来计算 dx_num
# eval_numerical_gradient_array 是一个函数，它计算指定函数的数值梯度
dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward_naive(x, pool_param)[0], x, dout)
# 进行最大池化的前向传播，得到输出 out 和缓存 cache
out, cache = max_pool_forward_naive(x, pool_param)
# 调用反向传播函数，计算梯度 dx
dx = max_pool_backward_naive(dout, cache)


# 输出的误差应该在 1e-12 左右
print('输出7：测试最大池化层的反向传播函数（输出梯度误差）')
print('Testing max_pool_backward_naive function:')
print('dx error: ', rel_error(dx, dx_num))


# # 快速层
# 使卷积和池化层变快可能会很具有挑战性。为了解决这个问题，我们提供了卷积和池化层的前向和反向传播的快速实现，
# 位于文件 `cs231n/fast_layers.py` 中。

# 快速卷积实现依赖于 Cython 扩展；要编译它，您需要在 `cs231n` 目录中运行以下命令：
#
# ```bash
# python setup.py build_ext --inplace
# ```

# 快速版本的卷积和池化层的 API 与您上面实现的朴素版本完全相同：前向传播接收数据、权重和参数，
# 并产生输出和缓存对象；反向传播接收上游导数和缓存对象，并产生相对于数据和权重的梯度。

# **注意：** 快速池化的实现只有在池化区域不重叠并且可以完全覆盖输入时才能表现出最佳性能。
# 如果不满足这些条件，则快速池化的实现不会比朴素实现快多少。

# 您可以通过运行以下代码比较朴素版本和快速版本的性能：
import pyximport
pyximport.install()

from cs231n.fast_layers import conv_forward_fast, conv_backward_fast
from time import time

# 创建随机输入数据 x，形状为 (100, 3, 31, 31)，表示 100 个样本，3 个通道，31x31 的特征图
x = np.random.randn(100, 3, 31, 31)
# 创建随机权重 w，形状为 (25, 3, 3, 3)，表示 25 个过滤器，每个过滤器的大小为 3x3
w = np.random.randn(25, 3, 3, 3)
# 创建随机偏置 b，形状为 (25,)
b = np.random.randn(25,)
# 创建随机上游导数 dout，形状为 (100, 25, 16, 16)，与卷积后的输出对应
dout = np.random.randn(100, 25, 16, 16)
# 定义卷积参数，包括步幅和填充
conv_param = {'stride': 2, 'pad': 1}

# 测量朴素卷积的前向传播时间
t0 = time()
out_naive, cache_naive = conv_forward_naive(x, w, b, conv_param) # 调用朴素卷积前向传播

# 测量快速卷积的前向传播时间
t1 = time()
out_fast, cache_fast = conv_forward_fast(x, w, b, conv_param) # 调用快速卷积前向传播
t2 = time()
print('输出8：卷积层前向传播朴素方法变到快速的时间和速度提升以及结果差异')
print('Testing conv_forward_fast:')
print('Naive: %fs' % (t1 - t0))
print('Fast: %fs' % (t2 - t1))
print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
print('Difference: ', rel_error(out_naive, out_fast))

# 测量朴素卷积和快速卷积的反向传播时间
t0 = time()
dx_naive, dw_naive, db_naive = conv_backward_naive(dout, cache_naive)  # 调用朴素卷积反向传播
t1 = time()
# 测量快速卷积的反向传播时间
dx_fast, dw_fast, db_fast = conv_backward_fast(dout, cache_fast)  # 调用快速卷积反向传播
t2 = time()

# 输出反向传播的时间和速度提升
print('输出9：卷积层反向传播朴素方法和快速方法的时间和速度差异&梯度差异')
print('\nTesting conv_backward_fast:')
print('Naive: %fs' % (t1 - t0)) # 朴素方法的时间
print('Fast: %fs' % (t2 - t1)) # 快速方法的时间
print('Speedup: %fx' % ((t1 - t0) / (t2 - t1))) # 计算速度提升
print('dx difference: ', rel_error(dx_naive, dx_fast)) # 输出输入梯度的差异
print('dw difference: ', rel_error(dw_naive, dw_fast)) # 输出权重梯度的差异
print('db difference: ', rel_error(db_naive, db_fast)) # 输出偏置梯度的差异

# 测试快速池化层
from cs231n.fast_layers import max_pool_forward_fast, max_pool_backward_fast

# 创建随机输入数据 x，形状为 (100, 3, 32, 32)
x = np.random.randn(100, 3, 32, 32)
# 创建随机上游导数 dout，形状为 (100, 3, 16, 16)
dout = np.random.randn(100, 3, 16, 16)
# 定义池化参数，包括池化高度、宽度和步幅
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

# 测量朴素池化的前向传播时间
t0 = time()
out_naive, cache_naive = max_pool_forward_naive(x, pool_param) # 调用朴素池化前向传播
t1 = time()
# 测量快速池化的前向传播时间
out_fast, cache_fast = max_pool_forward_fast(x, pool_param)  # 调用快速池化前向传播
t2 = time()

print('输出10：池化层前向传播的两种方法的差异')
print('Testing pool_forward_fast:')
print('Naive: %fs' % (t1 - t0))
print('fast: %fs' % (t2 - t1))
print('speedup: %fx' % ((t1 - t0) / (t2 - t1)))
print('difference: ', rel_error(out_naive, out_fast))

# 测量朴素池化的反向传播时间
t0 = time()
dx_naive = max_pool_backward_naive(dout, cache_naive)
t1 = time()
# 测量快速池化的反向传播时间
dx_fast = max_pool_backward_fast(dout, cache_fast)
t2 = time()

print('输出11：池化层反向传播两种方法的差异')
print('\nTesting pool_backward_fast:')
print('Naive: %fs' % (t1 - t0))
print('speedup: %fx' % ((t1 - t0) / (t2 - t1)))
print('dx difference: ', rel_error(dx_naive, dx_fast))


# # 卷积“夹心”层
# 我们之前介绍了“夹心”层的概念，它将多个操作组合成常用的模式。
# 在文件 `cs231n/layer_utils.py` 中，你会找到实现一些常用卷积网络模式的夹心层。

from cs231n.layer_utils import conv_relu_pool_forward, conv_relu_pool_backward

# 创建输入数据，形状为 (2, 3, 16, 16) 的随机张量
x = np.random.randn(2, 3, 16, 16)
w = np.random.randn(3, 3, 3, 3)  # 创建卷积核
b = np.random.randn(3,) # 创建偏置
dout = np.random.randn(2, 3, 8, 8) # 创建梯度输出
conv_param = {'stride': 1, 'pad': 1}  # 设置卷积参数
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2} # 设置池化参数

# 前向传播，计算输出和缓存
out, cache = conv_relu_pool_forward(x, w, b, conv_param, pool_param)
# 反向传播，计算梯度
dx, dw, db = conv_relu_pool_backward(dout, cache)

# 通过数值梯度评估反向传播的梯度
dx_num = eval_numerical_gradient_array(lambda x: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], b, dout)

print('输出12：测试“卷积 - ReLU - 池化”前向传播操作链（输出梯度误差）')
print('Testing conv_relu_pool')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

from cs231n.layer_utils import conv_relu_forward, conv_relu_backward

# 创建新的输入数据，形状为 (2, 3, 8, 8)
x = np.random.randn(2, 3, 8, 8)
w = np.random.randn(3, 3, 3, 3)
b = np.random.randn(3,)
dout = np.random.randn(2, 3, 8, 8)
conv_param = {'stride': 1, 'pad': 1}

# 前向传播，计算输出和缓存
out, cache = conv_relu_forward(x, w, b, conv_param)
# 反向传播，计算梯度
dx, dw, db = conv_relu_backward(dout, cache)

# 通过数值梯度评估反向传播的梯度
dx_num = eval_numerical_gradient_array(lambda x: conv_relu_forward(x, w, b, conv_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_relu_forward(x, w, b, conv_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_relu_forward(x, w, b, conv_param)[0], b, dout)

print('输出13：测试执行卷积后接ReLU激活函数是否正确（梯度误差）')
print('Testing conv_relu:')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))


# # 三层卷积网络
# 现在你已经实现了所有必要的层，我们可以将它们组合成一个简单的卷积网络。
#
# 打开文件 `cs231n/cnn.py` 并完成 `ThreeLayerConvNet` 类的实现。
# 运行以下单元格来帮助你调试：

# ## 健康检查损失
# 在构建新的网络后，你应该做的第一件事是检查损失。当我们使用 softmax 损失时，
# 我们期望随机权重（且没有正则化）的损失大约为 `log(C)`，其中 `C` 是类的数量。当我们添加正则化时，损失应该会上升。
model = ThreeLayerConvNet()

N = 50
X = np.random.randn(N, 3, 32, 32)
y = np.random.randint(10, size=N)
print('创建三层卷积网络后检查损失：')
loss, grads = model.loss(X, y)
print('输出14：无正则化的初始损失')
print('Initial loss (no regularization): ', loss)
# 设置正则化参数并重新计算损失
model.reg = 0.5
loss, grads = model.loss(X, y)
print('输出15：带有正则化的初始损失')
print('Initial loss (with regularization): ', loss)


# ## 梯度检查
# 在损失看起来合理后，使用数值梯度检查来确保反向传播是正确的。
# 当使用数值梯度检查时，应该使用少量人工数据和每层少量神经元。
num_inputs = 2  # 输入的样本数量
input_dim = (3, 16, 16)  # 输入的维度
reg = 0.0  # 正则化参数
num_classes = 10  # 类别数量
X = np.random.randn(num_inputs, *input_dim)
y = np.random.randint(num_classes, size=num_inputs)

# 初始化三层卷积网络
model = ThreeLayerConvNet(num_filters=3, filter_size=3,
                          input_dim=input_dim, hidden_dim=7,
                          dtype=np.float64)
loss, grads = model.loss(X, y)
# 对模型参数进行梯度检查
print('输出16：模型参数梯度检查结果')
for param_name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
    e = rel_error(param_grad_num, grads[param_name])
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))


# ## 过拟合小数据
# 一个不错的技巧是用少量训练样本来训练模型。你应该能够在小数据集上过拟合，
# 这将导致非常高的训练准确率和相对较低的验证准确率。
num_train = 100 # 训练样本数量
small_data = {
  'X_train': data['X_train'][:num_train],# 取前100个训练样本
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'], # 使用完整的验证集
  'y_val': data['y_val'],
}

# 初始化三层卷积网络
model = ThreeLayerConvNet(weight_scale=1e-2)

# 设置求解器
solver = Solver(model, small_data,
                num_epochs=20, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 5e-4,
                },
                verbose=True, print_every=1)
solver.train()


# 绘制损失、训练准确率和验证准确率的图表，以显示明显的过拟合：
print('输出17（图像）：可视化损失历史、训练集准确率以及验证集准确率')
plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


# ## 训练网络
# 通过训练三层卷积网络一个周期，你应该在训练集上达到超过40%的准确率：

# 初始化三层卷积网络，设置权重尺度、隐藏层维度和正则化
print('输出18：训练网络的过程')
model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

solver = Solver(model, data,
                num_epochs=1, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)  # 设置详细输出并每20次迭代打印一次
solver.train() # 开始训练


# ## 可视化滤波器
# 你可以通过运行以下代码可视化训练后网络的第一层卷积滤波器：
from cs231n.vis_utils import visualize_grid
print('输出19（图像）：可视化卷积滤波器')
# 转置并可视化第一层卷积滤波器
grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
plt.imshow(grid.astype('uint8'))
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()

# # 空间批量归一化
# 我们已经看到批量归一化是训练深度全连接网络的一个非常有用的技术。
# 批量归一化也可以用于卷积网络，但我们需要稍作修改；这种修改将被称为“空间批量归一化”。
#
# 通常情况下，批量归一化接受形状为`(N, D)`的输入，并产生形状为`(N, D)`的输出，其中我们在小批量维度`N`上进行归一化。
# 对于来自卷积层的数据，批量归一化需要接受形状为`(N, C, H, W)`的输入，并产生形状为`(N, C, H, W)`的输出，其中`N`维表示小批量大小，`(H, W)`维表示特征图的空间大小。
#
# 如果特征图是通过卷积生成的，那么我们期望每个特征通道的统计特性在不同图像和同一图像的不同位置之间相对一致。
# 因此，空间批量归一化通过在小批量维度`N`和空间维度`H`和`W`上计算统计量，来为每个`C`特征通道计算均值和方差。

# ## 空间批量归一化：前向传播
#
# 在文件`cs231n/layers.py`中，实现空间批量归一化的前向传播函数`spatial_batchnorm_forward`。通过运行以下代码检查你的实现：
# 检查训练时的前向传播，验证特征在空间批量归一化前后的均值和方差

N, C, H, W = 2, 3, 4, 5  # 设置批量大小、通道数、高度和宽度
x = 4 * np.random.randn(N, C, H, W) + 10  # 随机生成输入数据

print('输出20：空间批量归一化前后的形状、均值和标准差')
print('Before spatial batch normalization:')
print('  Shape: ', x.shape)
print('  Means: ', x.mean(axis=(0, 2, 3)))
print('  Stds: ', x.std(axis=(0, 2, 3)))

# 均值应接近0，标准差应接近1
gamma, beta = np.ones(C), np.zeros(C)
bn_param = {'mode': 'train'}
out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
print('After spatial batch normalization:')
print('  Shape: ', out.shape)
print('  Means: ', out.mean(axis=(0, 2, 3)))
print('  Stds: ', out.std(axis=(0, 2, 3)))

# Means should be close to beta and stds close to gamma
gamma, beta = np.asarray([3, 4, 5]), np.asarray([6, 7, 8])
out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
print('After spatial batch normalization (nontrivial gamma, beta):')
print('  Shape: ', out.shape)
print('  Means: ', out.mean(axis=(0, 2, 3)))
print('  Stds: ', out.std(axis=(0, 2, 3)))

# 检查测试时的前向传播，运行多次训练时的前向传播以热身运行平均值，
# 然后检查测试时前向传播后的激活均值和方差。

N, C, H, W = 10, 4, 11, 12 # 设置新的批量大小和特征图尺寸

bn_param = {'mode': 'train'} # 设置为训练模式
gamma = np.ones(C)
beta = np.zeros(C)
for t in range(50):  # 进行50次训练时前向传播
  x = 2.3 * np.random.randn(N, C, H, W) + 13  # 随机生成输入数据
  spatial_batchnorm_forward(x, gamma, beta, bn_param)  # 前向传播
bn_param['mode'] = 'test' # 切换到测试模式
x = 2.3 * np.random.randn(N, C, H, W) + 13 # 随机生成新的输入数据
a_norm, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param) # 测试时前向传播

# 均值应接近零，标准差应接近一，但会比训练时的前向传播更嘈杂。
print('After spatial batch normalization (test-time):')
print('  means: ', a_norm.mean(axis=(0, 2, 3)))
print('  stds: ', a_norm.std(axis=(0, 2, 3)))

# ## 空间批量归一化：反向传播
# 在文件`cs231n/layers.py`中，实现空间批量归一化的反向传播函数`spatial_batchnorm_backward`。
# 运行以下代码检查你的实现，使用数值梯度检查：

N, C, H, W = 2, 3, 4, 5
x = 5 * np.random.randn(N, C, H, W) + 12
gamma = np.random.randn(C)
beta = np.random.randn(C)
dout = np.random.randn(N, C, H, W)

bn_param = {'mode': 'train'}
fx = lambda x: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]
fg = lambda a: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]
fb = lambda b: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]

dx_num = eval_numerical_gradient_array(fx, x, dout)
da_num = eval_numerical_gradient_array(fg, gamma, dout)
db_num = eval_numerical_gradient_array(fb, beta, dout)

_, cache = spatial_batchnorm_forward(x, gamma, beta, bn_param)
dx, dgamma, dbeta = spatial_batchnorm_backward(dout, cache)
print('dx error: ', rel_error(dx_num, dx))
print('dgamma error: ', rel_error(da_num, dgamma))
print('dbeta error: ', rel_error(db_num, dbeta))


# #  实验！
# 进行实验，尝试在 CIFAR-10 上使用卷积网络（ConvNet）获得最佳性能。以下是一些建议，帮助你开始：
#
# ### 你可以尝试的内容：
# - 卷积核大小：上面我们使用了 7x7 的卷积核；虽然这能生成漂亮的图像，但较小的卷积核可能更高效。
# - 卷积核数量：上面我们使用了 32 个卷积核。更多或更少的卷积核哪个效果更好？
# - 批量归一化：尝试在卷积层之后添加空间批量归一化，在仿射层之后添加常规批量归一化。这样能加快网络的训练速度吗？
# - 网络架构：上面的网络有两层可训练参数。你能否通过更深的网络做得更好？
# 你可以在文件 cs231n/classifiers/convnet.py 中实现替代架构。可以尝试的一些良好架构包括：
#     - [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]
#     - [conv-relu-pool]XN - [affine]XM - [softmax or SVM]
#     - [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax or SVM]
#
# ### 训练提示
# 对于你尝试的每个网络架构，都应该调优学习率和正则化强度。在这样做时，有几点重要事项需要牢记：
# - 如果参数工作良好，你应该在几百次迭代内看到改进。
# - 记住逐步调整超参数的方法：首先测试一大范围的超参数，进行几次训练迭代，以找到有效的参数组合。
# - 一旦找到一些看起来有效的参数组合，围绕这些参数进行更细致的搜索。你可能需要训练更多的轮次。
# ### 超越常规
# 如果你感到冒险，可以实现许多其他功能来尝试提高性能。你不需要实现任何这些；但如果你愿意尝试，进行额外的探索是很好的选择。
# - 替代更新步骤：在这个任务中，我们实现了 SGD+动量、RMSprop 和 Adam；你可以尝试 AdaGrad 或 AdaDelta 等替代方法。
# - 替代激活函数，如 Leaky ReLU、Parametric ReLU 或 MaxOut。
# - 模型集成
# - 数据增强
#
# 如果你决定实现一些额外的内容，请在下面的“额外学分描述”单元中清楚描述它们，
# 并指向此文件或其他文件中的相关代码（如果适用）。
# ### 我们的期望
# 至少，你应该能够训练一个在验证集上获得至少 65% 准确率的 ConvNet。这只是一个下限——如果你细心操作，应该能够获得更高的准确率！对于特别高分的模型或独特的方法将给予额外学分。
# 你应该在下面的空间中进行实验并训练你的网络。该笔记本的最后一个单元应包含你最终训练网络的训练、验证和测试集的准确率。在这个笔记本中，你还应该写下你所做的工作，任何实现的额外功能，以及在训练和评估网络过程中制作的任何可视化或图表。
# 玩得开心，祝你训练顺利！
# 在 CIFAR-10 上训练一个优秀的模型