import numpy as np
from random import randrange

# 前面两个是用中心差分法计算单变量和多变量函数的数值梯度 微小增量h估计偏导
def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
  """ 
  计算 f 在 x 处的数值梯度的简单实现
  - f 应该是一个只接受一个参数的函数
  - x 是计算梯度的点（numpy数组）
  """ 

  fx = f(x)# 在原始点计算函数值
  grad = np.zeros_like(x)
  # 遍历 x 中的所有索引
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # 在 x+h 处计算函数
    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h # 增加 h
    fxph = f(x) # 计算 f(x + h)
    x[ix] = oldval - h
    fxmh = f(x) # 计算 f(x + h)
    x[ix] = oldval # 恢复

    # 使用中心差分公式计算偏导数
    grad[ix] = (fxph - fxmh) / (2 * h) # 斜率
    if verbose:
      print(ix, grad[ix])
    it.iternext() # 进入下一个维度

  return grad # 返回计算得到的梯度

def eval_numerical_gradient_array(f, x, df, h=1e-5):
  """
  为一个接受 numpy 数组并返回一个 numpy 数组的函数计算数值梯度。
  """
  grad = np.zeros_like(x) # 初始化梯度数组
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    ix = it.multi_index

    oldval = x[ix]  # 保存原值
    x[ix] = oldval + h  # 增加 h
    pos = f(x).copy()  # 计算 f(x + h)
    x[ix] = oldval - h  # 减少 h
    neg = f(x).copy()  # 计算 f(x - h)
    x[ix] = oldval  # 恢复原值

    # 计算梯度
    grad[ix] = np.sum((pos - neg) * df) / (2 * h)
    it.iternext() # 进入下一个维度
  return grad

def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
# 针对输入为多维数组和矩阵的情况
  """
  计算处理输入和输出 Blob 的函数的数值梯度。

    我们假设 f 接受多个输入 Blob 作为参数，最后是一个输出 Blob
    例如，f 的调用方式可能是：

    f(x, w, out)

    其中 x 和 w 是输入 Blob，f 的结果将写入 out。

    输入参数：
    - f: 函数
    - inputs: 输入 Blob 的元组
    - output: 输出 Blob
    - h: 步长
  """
  numeric_diffs = [] # 存储数值梯度
  for input_blob in inputs:
    diff = np.zeros_like(input_blob.diffs) # 初始化差分数组
    it = np.nditer(input_blob.vals, flags=['multi_index'],
                   op_flags=['readwrite'])
    while not it.finished:
      idx = it.multi_index
      orig = input_blob.vals[idx] # 保存原值

      input_blob.vals[idx] = orig + h
      f(*(inputs + (output,)))
      pos = np.copy(output.vals)
      input_blob.vals[idx] = orig - h
      f(*(inputs + (output,)))
      neg = np.copy(output.vals)
      input_blob.vals[idx] = orig

      #  计算数值差分
      diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

      it.iternext() # 进入下一个维度
    numeric_diffs.append(diff) # 添加到数值梯度列表
  return numeric_diffs # 返回所有的数值梯度


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
  # 计算网络的数值梯度
  return eval_numerical_gradient_blobs(lambda *args: net.forward(),
              inputs, output, h=h)


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
  """
  随机抽样几个元素，并仅返回这些维度的数值梯度。
  """

  for i in range(num_checks):
    ix = tuple([randrange(m) for m in x.shape])

    oldval = x[ix]  # 保存原值
    x[ix] = oldval + h  # 增加 h
    fxph = f(x)  # 计算 f(x + h)
    x[ix] = oldval - h  # 减少 h
    fxmh = f(x)  # 计算 f(x - h)
    x[ix] = oldval  # 恢复原值

    # 计算数值梯度
    grad_numerical = (fxph - fxmh) / (2 * h)
    grad_analytic = analytic_grad[ix]  # 解析梯度
    # 计算相对误差
    rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
    print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))

