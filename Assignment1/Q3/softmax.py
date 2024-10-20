import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
# 计算softmax损失及其梯度的朴素实现，使用显式的循环来遍历训练样本。
    """
       Softmax 损失函数，朴素实现（使用循环）

       输入的维度为 D，有 C 个类别，并且我们对 N 个示例的小批量进行操作。

       输入：
       - W: 形状为 (D, C) 的 numpy 数组，包含权重。
       - X: 形状为 (N, D) 的 numpy 数组，包含一个小批量的数据。
       - y: 形状为 (N,) 的 numpy 数组，包含训练标签；y[i] = c 意味着 X[i] 的标签为 c，其中 0 <= c < C。
       - reg: （浮点数）正则化强度

       返回一个元组：
       - 损失（单个浮点数）
       - 相对于权重 W 的梯度；与 W 形状相同的数组
       """
    # 初始化损失和梯度为零。
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: 使用显式循环计算 softmax 损失及其梯度。                     #
    # 将损失存储在 loss 中，将梯度存储在 dW 中。如果不小心               #
    # 在这里，很容易遇到数值不稳定性。别忘了正则化！                     #
    #############################################################################
    num_classes = W.shape[1] # 类别数量
    num_train = X.shape[0] # 训练样本数量
    # 通过循环遍历每个训练样本，计算其得分
    for i in xrange(num_train):
        scores = X[i].dot(W)  # 计算第 i 个样本的得分
        shift_scores = scores - max(scores)  # 减去最大值以提高数值稳定性
        loss_i = - shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))  # 计算损失
        loss += loss_i  # 累加损失
        for j in xrange(num_classes):
            softmax_output = np.exp(shift_scores[j]) / sum(np.exp(shift_scores))  # 计算 softmax 输出
            if j == y[i]:
                dW[:, j] += (-1 + softmax_output) * X[i]  # 更新对应类别的梯度
            else:
                dW[:, j] += softmax_output * X[i]  # 更新其他类别的梯度
    # 通过softmax公式计算损失和梯度，并累加；最后，计算平均损失并加入正则化项。
    loss /= num_train # 计算平均损失
    loss +=  0.5* reg * np.sum(W * W) # 加入正则化损失
    dW = dW/num_train + reg* W   # 计算平均梯度并加入正则化
    #pass
    #############################################################################
    #                          代码结束                                          #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
# 计算softmax损失及其梯度的向量化实现，避免使用显式循环。
# 向量化实现通常比朴素实现快得多，尤其是在处理大规模数据时。
  """
    Softmax 损失函数，向量化版本。

    输入和输出与 softmax_loss_naive 相同。
  """
  # 初始化损失和梯度为零。
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: 使用无显式循环计算 softmax 损失及其梯度。                     #
  # 将损失存储在 loss 中，将梯度存储在 dW 中。如果不小心               #
  # 在这里，很容易遇到数值不稳定性。别忘了正则化！                     #                                              #
  #############################################################################
# 使用矩阵运算一次性计算所有样本的得分和softmax输出，显著提高计算效率。
  num_classes = W.shape[1] # 类别数量
  num_train = X.shape[0] # 训练样本数量
  scores = X.dot(W) # 计算所有样本的得分
  shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1) # 减去最大值以提高数值稳定性
  # 计算 softmax 输出
  softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
  # 计算损失
  loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))
  # 计算平均损失
  loss /= num_train
  # 加入正则化损失
  loss +=  0.5* reg * np.sum(W * W)

  dS = softmax_output.copy()  # 复制 softmax 输出
  dS[range(num_train), list(y)] += -1 # 更新对应类别的梯度
  dW = (X.T).dot(dS) # 计算梯度
  dW = dW/num_train + reg* W  # 计算平均梯度并加入正则化
  #pass
  #############################################################################
  #                          代码结束                                          #
  #############################################################################

  return loss, dW

