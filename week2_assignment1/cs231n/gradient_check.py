import numpy as np
from random import randrange


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
  在点 x 处计算函数 f 的数值梯度的简单实现。
  - f 应该是一个接受单个参数的函数
  - x 是一个点（numpy 数组），用于评估梯度
  """

    fx = f(x)  # 在原始点评估函数值
    grad = np.zeros_like(x)  # 初始化梯度数组
    # 遍历 x 中的所有索引
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # 在 x + h 处评估函数
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # 增加 h
        fxph = f(x)  # 评估 f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # 评估 f(x - h)
        x[ix] = oldval  # 恢复原值

        # 使用中心差分公式计算偏导数
        grad[ix] = (fxph - fxmh) / (2 * h)  # 计算斜率
        if verbose:
            print(ix, grad[ix])  # 打印当前索引和梯度
        it.iternext()  # 移动到下一个维度

    return grad  # 返回计算得到的梯度


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
  为接受 numpy 数组并返回 numpy 数组的函数评估数值梯度。
  """
    grad = np.zeros_like(x)  # 初始化梯度数组
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h  # 增加 h
        pos = f(x).copy()  # 评估 f(x + h)
        x[ix] = oldval - h  # 减少 h
        neg = f(x).copy()  # 评估 f(x - h)
        x[ix] = oldval  # 恢复原值

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)  # 计算梯度
        it.iternext()  # 移动到下一个维度
    return grad  # 返回计算得到的梯度


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
  计算作用于输入和输出 blob 的数值梯度。

  我们假设 f 接受多个输入 blob 作为参数，最后是一个用于写入输出的 blob。
  例如，f 可能像这样被调用：

  f(x, w, out)

  其中 x 和 w 是输入 blob，f 的结果将写入 out。

  输入:
  - f: 函数
  - inputs: 输入 blob 的元组
  - output: 输出 blob
  - h: 步长
  """
    numeric_diffs = []  # 初始化数值差异列表
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)  # 初始化差异数组
        it = np.nditer(input_blob.vals, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]  # 保存原值

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))  # 计算 f
            pos = np.copy(output.vals)  # 复制输出
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))  # 计算 f
            neg = np.copy(output.vals)  # 复制输出
            input_blob.vals[idx] = orig  # 恢复原值

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)  # 计算数值梯度

            it.iternext()  # 移动到下一个维度
        numeric_diffs.append(diff)  # 添加到数值差异列表
    return numeric_diffs  # 返回所有差异


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    return eval_numerical_gradient_blobs(lambda *args: net.forward(), inputs, output, h=h)  # 对网络的前向传播计算数值梯度


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    随机抽样一些元素，仅在这些维度上返回数值梯度。
    """

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])  # 随机选择一个索引

        oldval = x[ix]
        x[ix] = oldval + h  # 增加 h
        fxph = f(x)  # 评估 f(x + h)
        x[ix] = oldval - h  # 减少 h
        fxmh = f(x)  # 评估 f(x - h)
        x[ix] = oldval  # 恢复原值

        grad_numerical = (fxph - fxmh) / (2 * h)  # 计算数值梯度
        grad_analytic = analytic_grad[ix]  # 获取解析梯度
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))  # 计算相对误差
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))  # 打印梯度和误差
