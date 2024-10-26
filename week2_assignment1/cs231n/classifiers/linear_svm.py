import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  # 简单的循环版本的结构化 SVM 损失计算
  """
  结构化 SVM 损失函数，朴素实现（使用循环）。

  输入的维度为 D，有 C 个类别，并且我们在 N 个示例的小批量上进行操作。

  输入：
  - W: 形状为 (D, C) 的 numpy 数组，包含权重。
  - X: 形状为 (N, D) 的 numpy 数组，包含一批数据。
  - y: 形状为 (N,) 的 numpy 数组，包含训练标签；y[i] = c 表示
    X[i] 的标签为 c，其中 0 <= c < C。
  - reg: （float）正则化强度

  返回一个元组：
  - 损失值，单个浮点数
  - 相对于权重 W 的梯度；与 W 形状相同的数组
  """
  dW = np.zeros(W.shape) # 初始化梯度为零

  # 计算损失和梯度
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  # 双重循环，首先计算每个训练样本的得分，然后计算每个类别的边际损失
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # 注意 delta = 1
      # 如果边际损失大于0，则将其加到总损失中，并更新相应类别的梯度。
      if margin > 0:
        loss += margin
        dW[:,j] += X[i].T
        dW[:,y[i]] += -X[i].T 

  # 目前的损失是所有训练示例的总和，但我们想要的是平均值，因此我们除以 num_train。
  loss /= num_train
  dW /= num_train
  # 向损失添加正则化。
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # 计算损失函数的梯度并将其存储在 dW 中。                                  #
  # 与其先计算损失再计算导数，不如在计算损失的同时计算导数，这可能会更简单。   #
  # 因此，你可能需要修改上面的代码来同时计算梯度。                          #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
    结构化 SVM 损失函数，向量化实现。

    输入和输出与 svm_loss_naive 相同。
  """
  loss = 0.0
  dW = np.zeros(W.shape) # 初始化梯度为零

  #############################################################################
  # TODO:                                                                     #
  # 实现结构化 SVM 损失的向量化版本，将结果存储在 loss 中。                   #
  #############################################################################
  # 通过矩阵运算计算所有训练样本的得分和边际损失，避免了显式的循环，使得计算更加高效
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  correct_class_scores = scores[list(range(num_train)), list(y)].reshape(-1,1) #(N, 1)
  margins = np.maximum(0, scores - correct_class_scores +1)
  margins[list(range(num_train)), list(y)] = 0
  loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)
  #pass

  #############################################################################
  # TODO:                                                                     #
  # 实现结构化 SVM 损失的梯度的向量化版本，将结果存储在 dW 中。              #
  #                                                                           #
  # 提示：与其从头计算梯度，不如重用一些你用于计算损失的中间值，可能会更容易。   #
  #############################################################################
  # 梯度计算通过构造一个系数矩阵 coeff_mat 来实现，这样可以同时处理多个样本，从而减少计算复杂度。
  coeff_mat = np.zeros((num_train, num_classes))
  coeff_mat[margins > 0] = 1
  coeff_mat[list(range(num_train)), list(y)] = 0
  coeff_mat[list(range(num_train)), list(y)] = -np.sum(coeff_mat, axis=1)

  dW = (X.T).dot(coeff_mat)
  dW = dW/num_train + reg*W
  #pass


  return loss, dW
