import numpy as np

# 执行仿射变换，计算输出out ；输入x被重塑为二维数组，便于与权重w相乘；
# 计算的结果是一个线性变换加上偏置
def affine_forward(x, w, b):
  """
  计算仿射（全连接）层的前向传播。

  输入 x 的形状为 (N, d_1, ..., d_k)，包含 N 个例子的一个小批量，
  其中每个例子 x[i] 的形状为 (d_1, ..., d_k)。我们将每个输入重塑为
  维度 D = d_1 * ... * d_k 的向量，然后将其转换为维度 M 的输出向量。

  输入：
  - x: 包含输入数据的 numpy 数组，形状为 (N, d_1, ..., d_k)
  - w: 权重的 numpy 数组，形状为 (D, M)
  - b: 偏置的 numpy 数组，形状为 (M,)

  返回一个元组：
  - out: 输出，形状为 (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: 实现仿射前向传播。将结果存储在 out 中。需要将输入重塑为行。                               #
  #############################################################################
  N = x.shape[0]  # 获取输入样本的数量 N
  x_rsp = x.reshape(N, -1)  # 将输入 x 重塑为形状 (N, D)，D 是特征数量
  out = x_rsp.dot(w) + b  # 计算输出：out = x * w + b

  cache = (x, w, b) # 缓存输入和权重、偏置以便于反向传播
  return out, cache

# 计算反向传播的梯度 计算关于输入x的梯度（dx）以及关于权重和偏置的梯度（dw 和 db）。
# dout：是从下一层传回的梯度
def affine_backward(dout, cache):
  """
  计算仿射层的反向传播。

  输入：
  - dout: 上游导数，形状为 (N, M)
  - cache: 元组：
    - x: 输入数据，形状为 (N, d_1, ... d_k)
    - w: 权重，形状为 (D, M)

  返回一个元组：
  - dx: 关于 x 的梯度，形状为 (N, d1, ..., d_k)
  - dw: 关于 w 的梯度，形状为 (D, M)
  - db: 关于 b 的梯度，形状为 (M,)
  """
  x, w, b = cache # 从缓存中解包输入 x、权重 w 和偏置 b
  dx, dw, db = None, None, None
  #############################################################################
  N = x.shape[0]  # 获取样本数量 N
  x_rsp = x.reshape(N, -1)  # 将输入 x 重塑为形状 (N, D)

  # 计算关于输入 x 的梯度
  dx = dout.dot(w.T)  # dout * w.T
  dx = dx.reshape(*x.shape)  # 将 dx 重塑为原始输入 x 的形状

  # 计算关于权重 w 和偏置 b 的梯度
  dw = x_rsp.T.dot(dout)  # x.T * dout
  db = np.sum(dout, axis=0)  # 对 dout 在样本维度上求和

  return dx, dw, db

# 执行 ReLU 激活函数的前向传播
# 计算结果是通过将所有负值设为 0 来修改输入 x。
def relu_forward(x):
  """
   计算修正线性单元（ReLU）层的前向传播。

  输入：
  - x: 任何形状的输入

  返回一个元组：
  - out: 输出，形状与 x 相同
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: 实现 ReLU 前向传播。                                   #
  #############################################################################
  out = x * (x >= 0) # 对每个元素进行 ReLU 操作：小于 0 的变为 0，大于等于 0 的保持不变

  cache = x # 缓存输入 x 以便于反向传播
  return out, cache

# 计算 ReLU 激活函数的反向传播 当x大于等于0，保持上游导数dout的值；否则设为0。实现ReLU的非线性
def relu_backward(dout, cache):
  """
  计算修正线性单元（ReLU）层的反向传播。

  输入：
  - dout: 上游导数，具有任意形状
  - cache: 输入 x，与 dout 形状相同

  返回：
  - dx: 关于 x 的梯度
  """
  dx, x = None, cache # 从缓存中获取输入 x
  #############################################################################
  # TODO: 实现 ReLU 的反向传播。                                   #
  #############################################################################
  dx = (x >= 0) * dout # 计算 dx，当 x 大于等于 0 时，保留 dout 的值，否则为 0

  return dx

# 实现批量归一化的前向传播，支持训练和测试模式
def batchnorm_forward(x, gamma, beta, bn_param):
  """
 批量归一化的前向传播。

  在训练过程中，从小批量统计数据计算样本均值和（未修正的）样本方差，
  并用于归一化输入数据。在训练过程中，我们还保留每个特征均值和方差
  的指数衰减运行平均，这些平均值在测试时用于归一化数据。

  在每个时间步中，我们使用基于动量参数的指数衰减更新均值和方差的运行平均：

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  请注意，批量归一化论文建议不同的测试时行为：他们使用大量训练图像计算每个特征的样本均值和方差
  而不是使用运行平均。对于此实现，我们选择使用运行平均，因为它们
  不需要额外的估计步骤；torch7 的批量归一化实现也使用运行平均。

  输入：
  - x: 形状为 (N, D) 的数据
  - gamma: 形状为 (D,) 的缩放参数
  - beta: 形状为 (D,) 的平移参数
  - bn_param: 字典，包含以下键：
    - mode: 'train' 或 'test'; 必需
    - eps: 数值稳定性的常量
    - momentum: 运行均值/方差的常量。
    - running_mean: 形状为 (D,) 的数组，给出特征的运行均值
    - running_var: 形状为 (D,) 的数组，给出特征的运行方差

  返回一个元组：
  - out: 形状为 (N, D)
  - cache: 需要在反向传播中使用的一组值
  """
  mode = bn_param['mode']  # 获取当前模式（训练或测试）
  eps = bn_param.get('eps', 1e-5)  # 获取数值稳定性的常量，默认为 1e-5
  momentum = bn_param.get('momentum', 0.9)  # 获取动量，默认为 0.9

  N, D = x.shape  # 获取输入的形状：N 是样本数量，D 是特征数量
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))  # 获取运行均值
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))  # 获取运行方差

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: 实现训练时批量归一化的前向传播。                                  #
    # 使用小批量统计量计算均值和方差，使用这些统计量归一化输入数据，并使用 gamma 和 beta 对
    # 归一化的数据进行缩放和平移。                                            #
    #                                                                           #
    # 您应将输出存储在 out 变量中。任何您需要在反向传播中使用的中间值应存储在 cache 变量中。 #
    #                                                                           #
    # 您还应使用计算得到的样本均值和方差结合动量变量来更新运行均值和运行方差，   #
    # 将结果存储在 running_mean 和 running_var 变量中。                    #
    #############################################################################
    sample_mean = np.mean(x, axis=0)  # 计算样本均值
    sample_var = np.var(x, axis=0)  # 计算样本方差
    x_hat = (x - sample_mean) / (np.sqrt(sample_var + eps))  # 归一化输入
    out = gamma * x_hat + beta  # 使用 gamma 和 beta 对归一化后的数据进行缩放和平移
    cache = (gamma, x, sample_mean, sample_var, eps, x_hat)  # 存储用于反向传播的中间值
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean  # 更新运行均值
    running_var = momentum * running_var + (1 - momentum) * sample_var  # 更新运行方差

  elif mode == 'test':
    #############################################################################
    # TODO: 实现测试时批量归一化的前向传播。                                   #
    # 使用运行均值和方差来归一化输入数据，然后使用 gamma 和 beta 对归一化后的数据进行缩放
    # 和偏移。将结果存储在 out 变量中。                                       #
    #############################################################################
    scale = gamma / (np.sqrt(running_var  + eps))  # 计算缩放因子
    out = x * scale + (beta - running_mean * scale)  # 使用运行均值和方差归一化输入

  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)  # 如果模式无效，抛出错误

  # 将更新后的运行均值存回 bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  批量归一化的反向传播。

  对于此实现，您应该在纸上写出批量归一化的计算图，并通过
  中间节点向后传播梯度。

  输入：
  - dout: 上游导数，形状为 (N, D)
  - cache: 从 batchnorm_forward 中获取的中间变量。

  返回一个元组：
  - dx: 关于输入 x 的梯度，形状为 (N, D)
  - dgamma: 关于缩放参数 gamma 的梯度，形状为 (D,)
  - dbeta: 关于平移参数 beta 的梯度，形状为 (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: 实现批量归一化的反向传播。将结果存储在 dx, dgamma 和 dbeta 变量中。                         #
  #############################################################################
  gamma, x, u_b, sigma_squared_b, eps, x_hat = cache # 从缓存中提取必要的中间变量
  N = x.shape[0]  # 获取样本数量 N

  dx_1 = gamma * dout  # 计算 x_hat 对 dout 的梯度
  dx_2_b = np.sum((x - u_b) * dx_1, axis=0)  # 计算关于均值的梯度
  dx_2_a = ((sigma_squared_b + eps) ** -0.5) * dx_1  # 计算归一化后对 dout 的梯度
  dx_3_b = (-0.5) * ((sigma_squared_b + eps) ** -1.5) * dx_2_b  # 计算关于方差的梯度
  dx_4_b = dx_3_b * 1  # 将 dx_3_b 复制
  dx_5_b = np.ones_like(x) / N * dx_4_b  # 计算与样本数相关的梯度
  dx_6_b = 2 * (x - u_b) * dx_5_b  # 计算归一化项的梯度
  dx_7_a = dx_6_b * 1 + dx_2_a * 1  # 将 dx_6_b 和 dx_2_a 结合
  dx_7_b = dx_6_b * 1 + dx_2_a * 1  # 再次将 dx_6_b 和 dx_2_a 结合
  dx_8_b = -1 * np.sum(dx_7_b, axis=0)  # 计算关于均值的反向梯度
  dx_9_b = np.ones_like(x) / N * dx_8_b  # 归一化关于均值的梯度
  dx_10 = dx_9_b + dx_7_a  # 组合梯度

  dgamma = np.sum(x_hat * dout, axis=0)  # 计算关于 gamma 的梯度
  dbeta = np.sum(dout, axis=0)  # 计算关于 beta 的梯度
  dx = dx_10  # 最终的输入梯度

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  批量归一化的替代反向传播。

  对于此实现，您应该在纸上计算批量归一化反向传播的导数，
  并尽可能简化。您应该能够为反向传播推导出一个简单的表达式。

  注意：此实现应期望接收与 batchnorm_backward 相同的缓存变量，
  但可能不会使用缓存中的所有值。

  输入/输出：与 batchnorm_backward 相同
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: 实现批量归一化的反向传播。将结果存储在 dx, dgamma 和 dbeta 变量中。 #
  #                                                                           #
  # 计算关于中心化输入的梯度后，您应该能够在一条语句中计算输入的梯度，  #
  # 我们的实现可以在一行 80 字符的行上完成。                             #
  #############################################################################
  gamma, x, sample_mean, sample_var, eps, x_hat = cache  # 从缓存中提取必要的中间变量
  N = x.shape[0]  # 获取样本数量 N
  dx_hat = dout * gamma  # 计算 x_hat 对 dout 的梯度
  dvar = np.sum(dx_hat * (x - sample_mean) * -0.5 * np.power(sample_var + eps, -1.5), axis=0)  # 关于方差的梯度
  dmean = np.sum(dx_hat * -1 / np.sqrt(sample_var + eps), axis=0) + dvar * np.mean(-2 * (x - sample_mean), axis=0)  # 关于均值的梯度
  dx = 1 / np.sqrt(sample_var + eps) * dx_hat + dvar * 2.0 / N * (x - sample_mean) + 1.0 / N * dmean  # 输入梯度
  dgamma = np.sum(x_hat * dout, axis=0)  # 计算关于 gamma 的梯度
  dbeta = np.sum(dout, axis=0)  # 计算关于 beta 的梯度

  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  执行（反向）dropout 的前向传播。

  输入：
  - x: 输入数据，任意形状
  - dropout_param: 一个字典，包含以下键：
    - p: Dropout 参数。我们以概率 p 丢弃每个神经元的输出。
    - mode: 'test' 或 'train'。如果模式为 train，则执行 dropout；
      如果模式为 test，则仅返回输入。
    - seed: 随机数生成器的种子。传递种子使该函数具有确定性，
      这对于梯度检查是必要的，但在真实网络中不是。

  输出：
  - out: 形状与 x 相同的数组。
  - cache: 元组 (dropout_param, mask)。在训练模式下，mask 是用于乘以输入的 dropout
    掩码；在测试模式下，mask 为 None。
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed']) # 设置随机种子以确保可重复性

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: 实现反向 dropout 的训练阶段前向传播。将 dropout 掩码存储在 mask 变量中。   #
    ###########################################################################
    mask = (np.random.rand(*x.shape) >= p) / (1 - p)
    #mask = (np.random.rand(x.shape[1]) >= p) / (1 - p)
    out = x * mask

  elif mode == 'test':
    ###########################################################################
    # TODO: 实现反向 dropout 的测试阶段前向传播。       #
    ###########################################################################
    out = x  # 在测试模式下，直接返回输入

  cache = (dropout_param, mask)  # 缓存参数和掩码
  out = out.astype(x.dtype, copy=False)  # 确保输出与输入具有相同的数据类型

  return out, cache


def dropout_backward(dout, cache):
  """
  执行（反向）dropout 的反向传播。

  输入：
  - dout: 上游导数，任意形状
  - cache: 来自 dropout_forward 的 (dropout_param, mask)。
  """
  dropout_param, mask = cache # 从缓存中获取参数和掩码
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: 实现反向 dropout 的训练阶段反向传播。   #
    ###########################################################################
    dx = dout * mask # 在训练模式下，应用掩码到上游导数

  elif mode == 'test':
    dx = dout  # 在测试模式下，直接返回上游导数
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  对卷积层的前向传播进行简单实现。

  输入包含 N 个数据点，每个数据点有 C 个通道，具有高度 H 和宽度 W。
  我们用 F 个不同的滤波器对每个输入进行卷积，每个滤波器覆盖所有 C 个通道，
  并具有高度 HH 和宽度 WW。

  输入：
  - x: 形状为 (N, C, H, W) 的输入数据
  - w: 形状为 (F, C, HH, WW) 的滤波器权重
  - b: 形状为 (F,) 的偏置
  - conv_param: 一个字典，包含以下键：
    - 'stride': 相邻感受野在水平和垂直方向之间的像素数量。
    - 'pad': 将用于零填充输入的像素数量。

  返回一个元组：
  - out: 形状为 (N, F, H', W') 的输出数据，其中 H' 和 W' 由以下公式给出：
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: 实现卷积的前向传播。                                              #
  # 提示：您可以使用 np.pad 函数进行填充。                                  #
  #############################################################################
  N, C, H, W = x.shape  # 获取输入的形状
  F, _, HH, WW = w.shape  # 获取滤波器的形状
  stride, pad = conv_param['stride'], conv_param['pad']  # 获取卷积参数
  # 计算输出的高度和宽度
  H_out = 1 + (H + 2 * pad - HH) // stride
  W_out = 1 + (W + 2 * pad - WW) // stride
  out = np.zeros((N, F, H_out, W_out))  # 初始化输出数组

  x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0) # 填充输入
  for i in range(H_out):
      for j in range(W_out):
          x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW] # 获取当前感受野
          for k in range(F):
              out[:, k , i, j] = np.sum(x_pad_masked * w[k, :, :, :], axis=(1,2,3)) # 计算卷积
          #out[:, : , i, j] = np.sum(x_pad_masked * w[:, :, :, :], axis=(1,2,3))
          
  #for k in range(F):
      #out[:, k, :, :] = out[:, k, :, :] + b[k]
  out = out + (b)[None, :, None, None]  # 添加偏置

  cache = (x, w, b, conv_param)  # 缓存输入、权重、偏置和卷积参数
  return out, cache


def conv_backward_naive(dout, cache):
  """
  卷积层的向后传播的简单实现。

    输入：
    - dout: 上游梯度。
    - cache: 一个元组 (x, w, b, conv_param)，与 conv_forward_naive 中相同。

    返回一个元组：
    - dx: 相对于 x 的梯度
    - dw: 相对于 w 的梯度
    - db: 相对于 b 的梯度
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: 实现卷积的反向传播。                          #
  #############################################################################
  x, w, b, conv_param = cache # 从缓存中提取输入、权重、偏置和卷积参数

  N, C, H, W = x.shape  # 输入的维度：样本数、通道数、高度和宽度
  F, _, HH, WW = w.shape  # 权重的维度：滤波器的数量、通道数、高度和宽度
  stride, pad = conv_param['stride'], conv_param['pad']  # 获取步幅和填充参数

  # 计算输出的高度和宽度
  H_out = 1 + (H + 2 * pad - HH) // stride
  W_out = 1 + (W + 2 * pad - WW) // stride

  # 对输入进行零填充
  x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
  dx = np.zeros_like(x) # 初始化 dx
  dx_pad = np.zeros_like(x_pad)  # 初始化填充后的 dx
  dw = np.zeros_like(w)  # 初始化 dw
  db = np.zeros_like(b)  # 初始化 db
  # 计算偏置的梯度
  db = np.sum(dout, axis = (0,2,3))

  # 计算 dw 和 dx_pad
  x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
  for i in range(H_out):
      for j in range(W_out):
          x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
          for k in range(F): # 计算 dw
              dw[k ,: ,: ,:] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis=0)
          for n in range(N): #计算 dx_pad
              dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += np.sum((w[:, :, :, :] * 
                                                 (dout[n, :, i, j])[:,None ,None, None]), axis=0)

  # 从填充后的梯度中去掉填充部分
  dx = dx_pad[:,:,pad:-pad,pad:-pad]

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  最大池化层的向前传播的简单实现。

    输入：
    - x: 输入数据，形状为 (N, C, H, W)
    - pool_param: 字典，包含以下键：
      - 'pool_height': 每个池化区域的高度
      - 'pool_width': 每个池化区域的宽度
      - 'stride': 相邻池化区域之间的距离

    返回一个元组：
    - out: 输出数据
    - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: 实现最大池化的向前传播                                #
  #############################################################################
  N, C, H, W = x.shape # 输入的维度
  HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']  # 获取池化参数

  # 计算输出的高度和宽度
  H_out = (H-HH)//stride+1
  W_out = (W-WW)//stride+1
  out = np.zeros((N,C,H_out,W_out))  # 初始化输出

  # 最大池化操作
  for i in range(H_out):
        for j in range(W_out):
            x_masked = x[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW]  # 提取当前池化区域
            out[:,:,i,j] = np.max(x_masked, axis=(2,3))  # 对每个通道取最大值


  cache = (x, pool_param)  # 缓存输入和池化参数
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  最大池化层的向后传播的简单实现。

    输入：
    - dout: 上游梯度
    - cache: 一个元组 (x, pool_param)，与向前传播中的相同。

    返回：
    - dx: 相对于 x 的梯度
  """
  dx = None
  #############################################################################
  # TODO: 实现最大池化的反向传播                                     #
  #############################################################################
  x, pool_param = cache  # 从缓存中提取输入和池化参数
  N, C, H, W = x.shape  # 输入的维度
  HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride'] # 获取池化参数

  # 计算输出的高度和宽度
  H_out = (H-HH)//stride+1
  W_out = (W-WW)//stride+1
  dx = np.zeros_like(x)  # 初始化 dx

  # 最大池化的反向传播
  for i in range(H_out):
     for j in range(W_out):
        x_masked = x[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW]  # 提取当前池化区域
        max_x_masked = np.max(x_masked,axis=(2,3))  # 在池化区域内找到最大值
        temp_binary_mask = (x_masked == (max_x_masked)[:,:,None,None])  # 创建一个二进制掩码
        # 仅在最大值位置传递梯度
        dx[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW] += temp_binary_mask * (dout[:,:,i,j])[:,:,None,None]

  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  计算空间批量归一化的向前传播。

    输入：
    - x: 形状为 (N, C, H, W) 的输入数据
    - gamma: 缩放参数，形状为 (C,)
    - beta: 移位参数，形状为 (C,)
    - bn_param: 字典，包含以下键：
      - mode: 'train' 或 'test'; 必需
      - eps: 数值稳定性的常数
      - momentum: 运行均值/方差的常数。momentum=0 表示每个时间步骤完全丢弃旧信息，而 momentum=1 表示永不合并新信息。默认为 momentum=0.9，在大多数情况下效果良好。
      - running_mean: 形状为 (D,) 的特征运行均值
      - running_var: 形状为 (D,) 的特征运行方差

    返回一个元组：
    - out: 形状为 (N, C, H, W) 的输出数据
    - cache: 反向传播所需的值
  """
  out, cache = None, None

  #############################################################################
  # TODO: 实现空间批量归一化的向前传播。                                   #
  #                                                                           #
  # 提示：可以使用上述定义的常规批量归一化来实现空间批量归一化。           #
  # 您的实现应该非常简短；我们的实现少于五行。                            #
  #############################################################################
  N, C, H, W = x.shape
  # 转换输入数据形状以适应批量归一化
  temp_output, cache = batchnorm_forward(x.transpose(0,3,2,1).reshape((N*H*W,C)), gamma, beta, bn_param)
  out = temp_output.reshape(N,W,H,C).transpose(0,3,2,1)  # 恢复输出的形状


  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  计算空间批量归一化的反向传播。

    输入：
    - dout: 上游梯度，形状为 (N, C, H, W)
    - cache: 从向前传播中获取的值

    返回一个元组：
    - dx: 相对于输入的梯度，形状为 (N, C, H, W)
    - dgamma: 相对于缩放参数的梯度，形状为 (C,)
    - dbeta: 相对于移位参数的梯度，形状为 (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: 实现空间批量归一化的反向传播。                                   #
  #                                                                           #
  # 提示：您可以使用上面定义的常规批量归一化来实现空间批量归一化。       #
  # 您的实现应该非常简短；我们的实现少于五行。                            #
  #############################################################################
  N,C,H,W = dout.shape
  dx_temp, dgamma, dbeta = batchnorm_backward_alt(dout.transpose(0,3,2,1).reshape((N*H*W,C)),cache)
  dx = dx_temp.reshape(N,W,H,C).transpose(0,3,2,1)

  return dx, dgamma, dbeta
  
# 计算SVM分类器的损失和梯度
def svm_loss(x, y):
  """
  使用多类 SVM 分类计算损失和梯度。

    输入：
    - x: 输入数据，形状为 (N, C)，其中 x[i, j] 是第 i 个输入的第 j 类的得分
    - y: 标签向量，形状为 (N,)，其中 y[i] 是 x[i] 的标签，并且 0 <= y[i] < C

    返回一个元组：
    - loss: 表示损失的标量
    - dx: 相对于 x 的损失梯度
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y] # 获取每个样本的正确类得分
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0) # 计算边际
  margins[np.arange(N), y] = 0 # 将正确类的边际设置为 0
  loss = np.sum(margins) / N # 计算平均损失
  num_pos = np.sum(margins > 0, axis=1) # 计算每个样本的正类数量
  dx = np.zeros_like(x) # 初始化 dx
  dx[margins > 0] = 1  # 对于边际大于 0 的样本，设置梯度为 1
  dx[np.arange(N), y] -= num_pos # 减去正类数量的梯度
  dx /= N # 平均化梯度
  return loss, dx

# 计算softmax分类器的损失和梯度
def softmax_loss(x, y):
  """
  计算软softmax分类器的损失和梯度。

    输入：
    - x: 输入数据，形状为 (N, C)，其中 x[i, j] 是第 i 个输入的第 j 类的得分
    - y: 标签向量，形状为 (N,)，其中 y[i] 是 x[i] 的标签，并且 0 <= y[i] < C

    返回一个元组：
    - loss: 表示损失的标量
    - dx: 相对于 x 的损失梯度
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True)) # 计算指数得分，防止溢出
  probs /= np.sum(probs, axis=1, keepdims=True) # 计算每个类的概率
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N  # 计算平均交叉熵损失
  dx = probs.copy()  # 复制概率作为梯度
  dx[np.arange(N), y] -= 1  # 对正确类的梯度减去 1
  dx /= N  # 平均化梯度
  return loss, dx
