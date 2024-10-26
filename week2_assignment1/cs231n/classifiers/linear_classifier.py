import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *

class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    使用随机梯度下降训练这个线性分类器。

        输入：
        - X: 形状为 (N, D) 的 numpy 数组，包含训练数据；有 N 个训练样本，每个样本的维度为 D。
        - y: 形状为 (N,) 的 numpy 数组，包含训练标签；y[i] = c 表示
          X[i] 的标签为 0 <= c < C，其中 C 为类别数。
        - learning_rate: （float）优化的学习率。
        - reg: （float）正则化强度。
        - num_iters: （整数）优化时的迭代步数。
        - batch_size: （整数）每一步使用的训练样本数量。
        - verbose: （布尔值）如果为真，则在优化过程中打印进度。

        输出：
        包含每次训练迭代中损失函数值的列表。
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # 懒惰地初始化 W
      self.W = 0.001 * np.random.randn(dim, num_classes)

    # 运行随机梯度下降以优化 W
    loss_history = []
    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO:                                                                 #
      # 从训练数据中随机抽取 batch_size 个样本及其对应的标签，用于这一轮的梯度下降。  #
      # 将数据存储在 X_batch 中，标签存储在 y_batch 中；抽样后 X_batch 应该具有
      # (dim, batch_size) 的形状，y_batch 应该具有 (batch_size,) 的形状。   #
      #                                                                       #
      # 提示：使用 np.random.choice 生成索引。带替换抽样的速度比不带替换抽样要快。   #
      #########################################################################
      batch_idx = np.random.choice(num_train, batch_size, replace = True)
      X_batch =  X[batch_idx]
      y_batch = y[batch_idx]
      #pass


      # 评估损失和梯度
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      # 执行参数更新
      #########################################################################
      # TODO:                                                                 #
      # 使用梯度和学习率更新权重。                                          #
      #########################################################################
      self.W += - learning_rate * grad
      # pass


      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
     使用训练好的线性分类器权重对数据点进行标签预测。

        输入：
        - X: D x N 的训练数据数组。每一列是一个 D 维点。

        返回：
        - y_pred: 对 X 中数据的预测标签。y_pred 是一个一维数组，长度为 N，每个元素是给定的预测类别的整数。
    """
    y_pred = np.zeros(X.shape[1])
    ###########################################################################
    # TODO:                                                                   #
    # 实现此方法。将预测标签存储在 y_pred 中。               #
    ###########################################################################
    scores = X.dot(self.W)
    y_pred = np.argmax(scores, axis = 1)
    #pass


    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    """
    计算损失函数及其导数。
           子类将覆盖此方法。

           输入：
           - X_batch: 形状为 (N, D) 的 numpy 数组，包含 N 个数据点的小批量；每个点的维度为 D。
           - y_batch: 形状为 (N,) 的 numpy 数组，包含小批量的标签。
           - reg: （float）正则化强度。

           返回：一个元组，包含：
           - 损失的单个浮点值
           - 相对于 self.W 的梯度；与 W 形状相同的数组
    """
    pass


class LinearSVM(LinearClassifier):
  """ 一个使用多类 SVM 损失函数的子类 """

  def loss(self, X_batch, y_batch, reg):
    # loss 方法调用了向量化的 SVM 损失函数 svm_loss_vectorized
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  # 也继承自 LinearClassifier，实现了 Softmax 损失的计算
  """ 一个使用 Softmax + 交叉熵损失函数的子类 """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

