import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  一个两层全连接神经网络。该网络的输入维度为 N，隐藏层维度为 H，并对 C 类进行分类。
     我们使用 softmax 损失函数和对权重矩阵的 L2 正则化来训练网络。该网络在第一个全连接层后使用 ReLU 非线性激活函数。

     换句话说，网络具有以下架构：

     输入 - 全连接层 - ReLU - 全连接层 - softmax

     第二个全连接层的输出是每个类的得分。
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
   初始化模型。权重被初始化为小的随机值，偏置初始化为零。
      权重和偏置存储在变量 self.params 中，它是一个字典，包含以下键：

      W1: 第一层权重; 形状为 (D, H)
      b1: 第一层偏置; 形状为 (H,)
      W2: 第二层权重; 形状为 (H, C)
      b2: 第二层偏置; 形状为 (C,)

      输入：
       - input_size: 输入数据的维度 D。
       - hidden_size: 隐藏层中的神经元数量 H。
       - output_size: 类别数量 C。
    """
    self.params = {}
    # 初始化第一层的权重 W1，形状为 (input_size, hidden_size)
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    # 初始化第一层的偏置 b1，形状为 (hidden_size,)
    self.params['b1'] = np.zeros(hidden_size)
    # 初始化第二层的权重 W2，形状为 (hidden_size, output_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    # 初始化第二层的偏置 b2，形状为 (output_size,)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    计算一个两层全连接神经网络的损失和梯度。

        输入：
        - X: 形状为 (N, D) 的输入数据。每个 X[i] 是一个训练样本。
        - y: 训练标签的向量。y[i] 是 X[i] 的标签，且每个 y[i] 是一个整数，范围在 0 <= y[i] < C。该参数是可选的；如果未传递，则仅返回分数，如果传递，则返回损失和梯度。
        - reg: 正则化强度。

        返回：
        如果 y 为 None，则返回形状为 (N, C) 的矩阵 scores，其中 scores[i, c] 是输入 X[i] 对于类别 c 的得分。

        如果 y 不为 None，则返回一个元组：
        - loss: 该批次训练样本的损失（数据损失和正则化损失）。
        - grads: 一个字典，将参数名称映射到相对于损失函数的这些参数的梯度；具有与 self.params 相同的键。
    """
    # 从 params 字典中解包变量
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # 计算前向传播
    scores = None
    #############################################################################
    # TODO: 执行前向传播，计算输入的类别得分。                             #
    # 将结果存储在 scores 变量中，该变量应为形状为 (N, C) 的数组。              #
    #############################################################################
    h_output = np.maximum(0, X.dot(W1) + b1) #(N,D) * (D,H) = (N,H)
    scores = h_output.dot(W2) + b2
    # pass


    # 如果未给出目标，则跳出，完成前向传播
    if y is None:
      return scores

    # 计算损失
    loss = None
    #############################################################################
    # TODO: 完成前向传播，并计算损失。该损失应包括                       #
    # W1 和 W2 的数据损失和 L2 正则化损失。将结果存储在损失变量中，              #
    # 它应该是一个标量。使用 Softmax 分类器的损失。为了使结果与我们匹配，          #
    # 将正则化损失乘以 0.5。                                                #
    #############################################################################
    # 为了数值稳定性，先平移得分
    shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)
    softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
    # 计算数据损失
    loss = -np.sum(np.log(softmax_output[list(range(N)), list(y)]))
    loss /= N # 平均损失
    # 添加正则化损失
    loss +=  0.5* reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    # pass




    # 反向传播：计算梯度
    grads = {}
    #############################################################################
    # TODO: 计算反向传播，计算权重和偏置的导数。将结果存储在 grads 字典中。 #
    # 例如， grads['W1'] 应该存储 W1 的梯度，并且是相同大小的矩阵。             #
    #############################################################################
    # pass
    dscores = softmax_output.copy()
    dscores[list(range(N)), list(y)] -= 1 # 计算梯度
    dscores /= N # 平均梯度

    # W2 和 b2 的梯度
    grads['W2'] = h_output.T.dot(dscores) + reg * W2
    grads['b2'] = np.sum(dscores, axis = 0)

    # 计算隐藏层的梯度
    dh = dscores.dot(W2.T)
    dh_ReLu = (h_output > 0) * dh # 反向传播通过 ReLU 激活

    # W1 和 b1 的梯度
    grads['W1'] = X.T.dot(dh_ReLu) + reg * W1
    grads['b1'] = np.sum(dh_ReLu, axis = 0)
    

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    # 用于训练神经网络。它使用随机梯度下降（SGD）方法来更新网络的权重和偏置。
    """
    使用随机梯度下降训练这个神经网络。

            输入：
            - X: 形状为 (N, D) 的 numpy 数组，给出训练数据。
            - y: 形状为 (N,) 的 numpy 数组，给出训练标签；y[i] = c 表示 X[i] 的标签为 c，且 0 <= c < C。
            - X_val: 形状为 (N_val, D) 的 numpy 数组，给出验证数据。
            - y_val: 形状为 (N_val,) 的 numpy 数组，给出验证标签。
            - learning_rate: 优化的学习率标量。
            - learning_rate_decay: 一个标量，给出每个周期后用于衰减学习率的因子。
            - reg: 标量，给出正则化强度。
            - num_iters: 优化的步数。
            - batch_size: 每步使用的训练样本数量。
            - verbose: 布尔值；如果为真，则在优化过程中打印进度。
    """
    # 每个周期内，随机抽取一个小批量的训练数据和标签，计算损失及其梯度，然后更新模型的参数。
    # 训练样本数量
    num_train = X.shape[0]
    # 每个周期的迭代次数
    iterations_per_epoch = max(num_train / batch_size, 1)

    # 使用 SGD 优化 self.model 中的参数
    loss_history = []  # 记录损失历史
    train_acc_history = []  # 记录训练准确率历史
    val_acc_history = []  # 记录验证准确率历史

    for it in range(num_iters):
      # 注意：使用 range 替代 xrange，以兼容 Python 3
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: 创建一个随机的小批量训练数据和标签，将它们分别存储在 X_batch 和 y_batch 中。 #
      #########################################################################
      idx = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[idx]
      y_batch = y[idx]
      # pass


      # 使用当前小批量计算损失和梯度
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      # 记录当前损失
      loss_history.append(loss)

      #########################################################################
      # TODO: 使用 grads 字典中的梯度来更新网络参数（存储在字典 self.params 中）， #
      # 使用随机梯度下降。您需要使用上述定义的 grads 字典中的梯度。       #
      #########################################################################
      # 使用 SGD 更新权重和偏置
      self.params['W2'] += - learning_rate * grads['W2']
      self.params['b2'] += - learning_rate * grads['b2']
      self.params['W1'] += - learning_rate * grads['W1']
      self.params['b1'] += - learning_rate * grads['b1']
      # pass

      # 如果 verbose 为真，并且当前迭代次数是 100 的倍数，则打印进度
      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # 每个周期检查训练和验证准确率，并衰减学习率
      if it % iterations_per_epoch == 0:
        # 检查训练准确率
        train_acc = (self.predict(X_batch) == y_batch).mean()
        # 检查验证准确率
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # 衰减学习率
        learning_rate *= learning_rate_decay
    # 返回损失历史、训练准确率历史和验证准确率历史
    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    # 用于根据训练好的权重预测输入数据的标签。
    # 先计算隐藏层的输出（经过 ReLU 激活），然后计算输出层的得分，最后返回得分最高的类别作为预测标签。
    """
    使用训练好的权重对数据点进行标签预测。对于每个数据点，我们预测每个 C 类的得分，并将每个数据点分配给得分最高的类。

        输入：
        - X: 形状为 (N, D) 的 numpy 数组，给出 N 个 D 维数据点以进行分类。

        返回：
        - y_pred: 形状为 (N,) 的 numpy 数组，给出每个元素的预测标签。
        对于所有 i，y_pred[i] = c 表示 X[i] 被预测为类 c，其中 0 <= c < C。
    """
    y_pred = None

    ###########################################################################
    # TODO: 实现这个函数；它应该是非常简单的！                #
    ###########################################################################
    # 计算隐藏层的输出，使用 ReLU 激活函数
    h = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
    scores = h.dot(self.params['W2']) + self.params['b2']
    y_pred = np.argmax(scores, axis=1)
    # pass

    return y_pred


