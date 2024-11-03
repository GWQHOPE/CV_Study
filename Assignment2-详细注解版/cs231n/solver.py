import numpy as np

from cs231n import optim


class Solver(object):
  """
  Solver类封装了训练分类模型所需的所有逻辑。Solver执行使用optim.py中定义的不同更新规则的随机梯度下降。

  该求解器接受训练和验证数据及标签，因此可以定期检查训练和验证数据上的分类准确性，以监控过拟合。

  要训练模型，您首先需要构造一个Solver实例，将模型、数据集和各种选项（学习率、批量大小等）传递给构造函数。然后，您可以调用train()方法来运行优化过程并训练模型。

  train()方法返回后，model.params将包含在训练过程中在验证集上表现最佳的参数。此外，实例变量solver.loss_history将包含训练过程中遇到的所有损失列表，实例变量solver.train_acc_history和solver.val_acc_history将包含模型在每个epoch上的训练和验证集的准确率。

  示例用法可能如下所示：

  data = {
    'X_train': # 训练数据
    'y_train': # 训练标签
    'X_val': # 验证数据
    'y_val': # 验证标签
  }
  model = MyAwesomeModel(hidden_size=100, reg=10)
  solver = Solver(model, data,
                  update_rule='sgd',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  lr_decay=0.95,
                  num_epochs=10, batch_size=100,
                  print_every=100)
  solver.train()


   Solver 适用于一个符合以下 API 的模型对象：

  - model.params 必须是一个字典，将字符串参数名称映射到包含参数值的 numpy 数组。

  - model.loss(X, y) 必须是一个函数，计算训练时的损失和梯度，及测试时的分类得分，具有以下输入和输出：

    输入：
    - X: 给定形状为 (N, d_1, ..., d_k) 的输入数据的一个小批量数组
    - y: 标签数组，形状为 (N,) ，y[i] 是 X[i] 的标签。

    返回值：
    如果 y 为 None，则运行测试时前向传播并返回：
    - scores: 形状为 (N, C) 的分类得分数组，其中 scores[i, c] 表示 X[i] 对应类别 c 的得分。

    如果 y 不为 None，则运行训练时的前向和反向传播，并返回一个元组：
    - loss: 一个标量，表示损失
    - grads: 一个字典，具有与 self.params 相同的键，将参数名称映射到损失相对于这些参数的梯度。
  """

  def __init__(self, model, data, **kwargs):
    """
    构造一个新的 Solver 实例。

    必要参数：
    - model: 一个符合上述 API 的模型对象
    - data: 包含训练和验证数据的字典，格式如下：
      'X_train': 形状为 (N_train, d_1, ..., d_k) 的训练图像数组
      'X_val': 形状为 (N_val, d_1, ..., d_k) 的验证图像数组
      'y_train': 形状为 (N_train,) 的训练图像标签
      'y_val': 形状为 (N_val,) 的验证图像标签

    可选参数：
    - update_rule: 表示 optim.py 中更新规则的字符串名称。默认值为 'sgd'。
    - optim_config: 包含超参数的字典，将传递给选定的更新规则。每种更新规则需要不同的超参数（参见 optim.py），但所有更新规则都需要一个 'learning_rate' 参数，因此该参数必须始终存在。
    - lr_decay: 学习率衰减的标量；每个 epoch 后学习率乘以此值。
    - batch_size: 用于计算损失和梯度的小批量大小。
    - num_epochs: 训练时运行的 epoch 数。
    - print_every: 整数值；每隔 print_every 次迭代打印训练损失。
    - verbose: 布尔值；若为 False，则训练期间不打印输出。
    """
    self.model = model # 模型实例
    self.X_train = data['X_train']
    self.y_train = data['y_train']
    self.X_val = data['X_val']
    self.y_val = data['y_val']
    
    # 解包关键字参数
    self.update_rule = kwargs.pop('update_rule', 'sgd')  # 更新规则，默认为'sgd'
    self.optim_config = kwargs.pop('optim_config', {})  # 优化配置
    self.lr_decay = kwargs.pop('lr_decay', 1.0)  # 学习率衰减，默认为1.0
    self.batch_size = kwargs.pop('batch_size', 100)  # 批量大小，默认为100
    self.num_epochs = kwargs.pop('num_epochs', 10)  # epoch数量，默认为10

    self.print_every = kwargs.pop('print_every', 10)  # 打印频率，默认为10
    self.verbose = kwargs.pop('verbose', True)  # 是否详细输出，默认为True

    # 如果有额外的关键字参数，则抛出错误
    if len(kwargs) > 0:
      extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
      raise ValueError('Unrecognized arguments %s' % extra)

    # Make sure the update rule exists, then replace the string
    # name with the actual function
    if not hasattr(optim, self.update_rule):
      raise ValueError('Invalid update_rule "%s"' % self.update_rule)
    self.update_rule = getattr(optim, self.update_rule)

    self._reset()


  def _reset(self):
    """
    设置一些优化的书面变量。不要手动调用此方法。
    """
    # 设置一些变量用于记录
    self.epoch = 0  # 当前epoch
    self.best_val_acc = 0  # 最佳验证准确率
    self.best_params = {}  # 最佳参数
    self.loss_history = []  # 损失历史记录
    self.train_acc_history = []  # 训练准确率历史记录
    self.val_acc_history = []  # 验证准确率历史记录

    # 为每个参数深拷贝优化配置
    self.optim_configs = {}
    for p in self.model.params:
      d = {k: v for k, v in self.optim_config.items()} # 复制当前优化配置
      self.optim_configs[p] = d  # 存储每个参数的配置


  def _step(self):
    """
    执行一次梯度更新。此方法由train()调用，不应手动调用。
    """
    # 制作一小批训练数据
    num_train = self.X_train.shape[0]  # 训练数据数量
    batch_mask = np.random.choice(num_train, self.batch_size)  # 随机选择批量样本
    X_batch = self.X_train[batch_mask]  # 批量数据
    y_batch = self.y_train[batch_mask]  # 批量标签

    # 计算损失和梯度
    loss, grads = self.model.loss(X_batch, y_batch)  # 计算损失和梯度
    self.loss_history.append(loss)  # 记录损失

    # 执行参数更新
    for p, w in self.model.params.items():  # 遍历模型参数
      dw = grads[p]  # 获取梯度
      config = self.optim_configs[p]  # 获取当前参数的优化配置
      next_w, next_config = self.update_rule(w, dw, config)  # 更新参数
      self.model.params[p] = next_w  # 更新模型参数
      self.optim_configs[p] = next_config  # 更新优化配置


  def check_accuracy(self, X, y, num_samples=None, batch_size=100):
    """
    检查模型在提供数据上的准确性。

    输入：
    - X: 形状为(N, d_1, ..., d_k)的数据数组
    - y: 形状为(N,)的标签数组
    - num_samples: 如果不是None，则对数据进行子采样，仅在num_samples个数据点上测试模型。
    - batch_size: 将X和y拆分为该大小的批次，以避免占用过多内存。

    返回：
    - acc: 标量，给出模型正确分类的实例的比例。
    """

    # 可能进行子采样
    N = X.shape[0]
    if num_samples is not None and N > num_samples:
      mask = np.random.choice(N, num_samples) # 随机选择样本
      N = num_samples
      X = X[mask]
      y = y[mask]

    # 按批次计算预测
    num_batches = N / batch_size
    if N % batch_size != 0:
      num_batches += 1
    y_pred = []
    for i in range(int(num_batches)):
      start = i * batch_size
      end = (i + 1) * batch_size
      scores = self.model.loss(X[start:end]) # 计算当前批次的得分
      y_pred.append(np.argmax(scores, axis=1))  # 获取每个样本的预测类别
    y_pred = np.hstack(y_pred) # 将所有批次的预测结果合并
    acc = np.mean(y_pred == y) # 计算准确率

    return acc


  def train(self):
    """
    执行优化以训练模型。
    """
    num_train = self.X_train.shape[0]  # 训练数据的数量
    iterations_per_epoch = max(num_train / self.batch_size, 1) # 每个epoch的迭代次数
    num_iterations = self.num_epochs * iterations_per_epoch # 总的迭代次数

    for t in range(int(num_iterations)):
      self._step() # 执行一次梯度更新

      # 可能打印训练损失
      if self.verbose and t % self.print_every == 0:
        print('(Iteration %d / %d) loss: %f' % (
               t + 1, num_iterations, self.loss_history[-1]))

      # 在每个epoch（遍历整个训练数据集一次的过程）结束时，增加epoch计数器并衰减学习率。
      epoch_end = (t + 1) % iterations_per_epoch == 0
      if epoch_end:
        self.epoch += 1
        for k in self.optim_configs:
          self.optim_configs[k]['learning_rate'] *= self.lr_decay  # 衰减学习率

      # 在第一次迭代、最后一次迭代和每个epoch结束时检查训练和验证准确率。
      first_it = (t == 0)
      last_it = (t == num_iterations + 1)
      if first_it or last_it or epoch_end:
        train_acc = self.check_accuracy(self.X_train, self.y_train,
                                        num_samples=1000)  # 计算训练准确率
        val_acc = self.check_accuracy(self.X_val, self.y_val)  # 计算验证准确率
        self.train_acc_history.append(train_acc)  # 记录训练准确率
        self.val_acc_history.append(val_acc) # 记录验证准确率

        if self.verbose:
          print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                 self.epoch, self.num_epochs, train_acc, val_acc))

        # 记录最佳模型
        if val_acc > self.best_val_acc:
          self.best_val_acc = val_acc
          self.best_params = {}
          for k, v in self.model.params.items():
            self.best_params[k] = v.copy() # 复制最佳参数

    # 在训练结束时将最佳参数替换到模型中
    self.model.params = self.best_params

