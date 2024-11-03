from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  便利层，执行仿射变换后接ReLU激活

  输入：
  - x: 输入到仿射层
  - w, b: 仿射层的权重和偏置

  返回值：
  - out: ReLU的输出
  - cache: 用于反向传播的对象
  """
  a, fc_cache = affine_forward(x, w, b) # 执行仿射前向传播
  out, relu_cache = relu_forward(a) # 执行ReLU前向传播
  cache = (fc_cache, relu_cache) # 缓存前向传播的中间结果
  return out, cache


def affine_relu_backward(dout, cache):
  """
  仿射-ReLU便利层的反向传播
  """
  fc_cache, relu_cache = cache  # 从缓存中提取中间结果
  da = relu_backward(dout, relu_cache)  # 执行ReLU的反向传播
  dx, dw, db = affine_backward(da, fc_cache)  # 执行仿射层的反向传播
  return dx, dw, db  # 返回输入的梯度和权重、偏置的梯度


pass

def affine_bn_relu_forward(x , w , b, gamma, beta, bn_param):
    """
        便利层，执行仿射变换后接批归一化和ReLU激活

        输入：
        - x: 输入到仿射层
        - w, b: 仿射层的权重和偏置
        - gamma, beta: 批归一化的缩放和偏移参数
        - bn_param: 批归一化的参数

        返回值：
        - out: ReLU的输出
        - cache: 用于反向传播的对象
    """
    a, fc_cache = affine_forward(x, w, b)  # 执行仿射前向传播
    bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)  # 执行批归一化
    out, relu_cache = relu_forward(bn)  # 执行ReLU前向传播
    cache = (fc_cache, bn_cache, relu_cache)  # 缓存前向传播的中间结果
    return out, cache

def affine_bn_relu_backward(dout, cache):
    """
        批归一化-ReLU便利层的反向传播
    """
    fc_cache, bn_cache, relu_cache = cache  # 从缓存中提取中间结果
    dbn = relu_backward(dout, relu_cache)  # 执行ReLU的反向传播
    da, dgamma, dbeta = batchnorm_backward_alt(dbn, bn_cache)  # 执行批归一化的反向传播
    dx, dw, db = affine_backward(da, fc_cache)  # 执行仿射层的反向传播
    return dx, dw, db, dgamma, dbeta  # 返回输入的梯度和权重、偏置、批归一化参数的梯度


def conv_relu_forward(x, w, b, conv_param):
  """
  便利层，执行卷积后接ReLU激活。

  输入：
  - x: 输入到卷积层
  - w, b, conv_param: 卷积层的权重和参数

  返回值：
  - out: ReLU的输出
  - cache: 用于反向传播的对象
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)  # 执行卷积前向传播
  out, relu_cache = relu_forward(a)  # 执行ReLU前向传播
  cache = (conv_cache, relu_cache)  # 缓存前向传播的中间结果
  return out, cache


def conv_relu_backward(dout, cache):
  """
  卷积-ReLU便利层的反向传播。
  """
  conv_cache, relu_cache = cache # 从缓存中提取中间结果
  da = relu_backward(dout, relu_cache) # 执行ReLU的反向传播
  dx, dw, db = conv_backward_fast(da, conv_cache) # 执行卷积层的反向传播
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  便利层，执行卷积、ReLU和池化。

  输入：
  - x: 输入到卷积层
  - w, b, conv_param: 卷积层的权重和参数
  - pool_param: 池化层的参数

  返回值：
  - out: 池化层的输出
  - cache: 用于反向传播的对象
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)  # 执行卷积前向传播
  s, relu_cache = relu_forward(a)  # 执行ReLU前向传播
  out, pool_cache = max_pool_forward_fast(s, pool_param)  # 执行池化前向传播
  cache = (conv_cache, relu_cache, pool_cache)  # 缓存前向传播的中间结果
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  卷积-ReLU-池化便利层的反向传播
  """
  conv_cache, relu_cache, pool_cache = cache  # 从缓存中提取中间结果
  ds = max_pool_backward_fast(dout, pool_cache)  # 执行池化的反向传播
  da = relu_backward(ds, relu_cache)  # 执行ReLU的反向传播
  dx, dw, db = conv_backward_fast(da, conv_cache)  # 执行卷积层的反向传播
  return dx, dw, db  # 返回输入的梯度和权重、偏置的梯度

