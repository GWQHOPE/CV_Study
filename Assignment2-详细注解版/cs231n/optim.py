import numpy as np

"""
该文件实现了多种常用的第一阶更新规则，用于训练神经网络。每个更新规则接受当前权重和损失相对于这些权重的梯度，并生成下一组权重。每个更新规则都有相同的接口：

def update(w, dw, config=None):

输入：
  - w: 一个 numpy 数组，表示当前的权重。
  - dw: 一个与 w 形状相同的 numpy 数组，表示损失相对于 w 的梯度。
  - config: 一个字典，包含超参数值，如学习率、动量等。如果更新规则需要在多次迭代中缓存值，那么 config 还将保存这些缓存值。

返回：
  - next_w: 更新后的下一个权重。
  - config: config 字典，用于传递给下一次迭代的更新规则。

注意：对于大多数更新规则，默认学习率可能表现不佳；然而其他超参数的默认值应该适用于各种不同的问题。

为了提高效率，更新规则可以进行就地更新，改变 w 的值并将 next_w 设置为 w。
"""

# 随机梯度下降：直接根据当前的梯度和学习率更新权重。
def sgd(w, dw, config=None):
  """
  执行简单的随机梯度下降。

  config 格式：
  - learning_rate: 标量学习率。
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)

  w -= config['learning_rate'] * dw
  return w, config

# 带动量的随机梯度下降 使用前几次的梯度的加权平均来更新权重，这样可以加快收敛速度并减少震荡。
def sgd_momentum(w, dw, config=None):
  """
  执行带动量的随机梯度下降。

  config 格式：
  - learning_rate: 标量学习率。
  - momentum: 一个介于 0 和 1 之间的标量，表示动量值。
    设置 momentum = 0 则退化为简单的 SGD。
  - velocity: 与 w 和 dw 形状相同的 numpy 数组，用于存储梯度的移动平均。
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)
  v = config.get('velocity', np.zeros_like(w))
  
  next_w = None
  #############################################################################
  # TODO: 实现动量更新公式。将更新后的值存储在 next_w 变量中。您还应使用并更新速度 v。       #
  #############################################################################
  v = config['momentum'] * v - config['learning_rate'] * dw
  next_w = w + v

  config['velocity'] = v

  return next_w, config

# 通过计算梯度的平方的移动平均，为每个参数设置自适应学习率，
# 解决了 SGD 在面对不同尺度参数时学习率调整不灵活的问题。
def rmsprop(x, dx, config=None):
  """
  使用 RMSProp 更新规则，该规则利用梯度平方值的移动平均来设置自适应的每参数学习率。

  config 格式：
  - learning_rate: 标量学习率。
  - decay_rate: 一个介于 0 和 1 之间的标量，给出平方梯度缓存的衰减率。
  - epsilon: 小的标量，用于平滑，避免除以零。
  - cache: 梯度的二次矩移动平均。
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('decay_rate', 0.99)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('cache', np.zeros_like(x))

  next_x = None
  #############################################################################
  # TODO: 实现 RMSprop 更新公式，将 x 的下一个值存储在 next_x 变量中。不要忘记更新存储在      #
  # config['cache'] 中的缓存值。                                                        #
  #############################################################################
  config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * (dx**2)
  next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']) + config['epsilon'])

  return next_x, config

# 结合了 RMSProp 和带动量的 SGD 的优点，使用梯度的一阶矩（平均值）和二阶矩（平方的平均值），
# 同时引入偏差校正，有助于更稳定和快速的收敛。
def adam(x, dx, config=None):
  """
  使用 Adam 更新规则，该规则结合了梯度及其平方的移动平均和偏差校正项。

  config 格式：
  - learning_rate: 标量学习率。
  - beta1: 一阶矩的移动平均的衰减率。
  - beta2: 二阶矩的移动平均的衰减率。
  - epsilon: 小的标量，用于平滑，避免除以零。
  - m: 梯度的移动平均。
  - v: 梯度平方的移动平均。
  - t: 迭代次数。
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(x))
  config.setdefault('v', np.zeros_like(x))
  config.setdefault('t', 0)
  
  next_x = None
  #############################################################################
  # TODO: 实现 Adam 更新公式，将 x 的下一个值存储在 next_x 变量中。不要忘记更新存储在 m, v, 和 t 中的变量。     #
  #############################################################################
  config['t'] += 1
  config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
  config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dx**2)
  mb = config['m'] / (1 - config['beta1']**config['t'])
  vb = config['v'] / (1 - config['beta2']**config['t'])
  next_x = x - config['learning_rate'] * mb / (np.sqrt(vb) + config['epsilon'])

  return next_x, config