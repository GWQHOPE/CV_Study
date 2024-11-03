import numpy as np
from cs231n.layers import *
from cs231n.layer_utils import *
class TwoLayerNet(object):
    """
  一个具有 ReLU 非线性激活和 softmax 损失的两层全连接神经网络，使用模块化的层设计。
  我们假设输入维度为 D，隐藏层维度为 H，并进行 C 类分类。

  网络结构应该为：仿射 - ReLU - 仿射 - softmax。

  注意：该类不实现梯度下降；它会与一个单独的 Solver 对象交互，由 Solver 负责进行优化。

  模型的可学习参数存储在字典 self.params 中，该字典将参数名称映射到 numpy 数组。
  """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
    初始化一个新的网络。

    参数:
    - input_dim: 一个整数，表示输入的大小
    - hidden_dim: 一个整数，表示隐藏层的大小
    - num_classes: 一个整数，表示分类的类别数
    - dropout: 介于 0 和 1 之间的数值，表示 dropout 的强度
    - weight_scale: 一个标量，表示权重随机初始化的标准差
    - reg: 一个标量，表示 L2 正则化的强度
    """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: 初始化两层网络的权重和偏置。权重应从标准差等于 weight_scale 的高斯分布中初始化， #
        # 偏置应初始化为零。所有的权重和偏置都应存储在字典 self.params 中，第一层的权重和偏置         #
        # 使用键 'W1' 和 'b1'，第二层的权重和偏置使用键 'W2' 和 'b2'。                       #
        ############################################################################
        # 初始化第一层权重和偏置
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        # 初始化第二层权重
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        # pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
            计算一个小批量数据的损失和梯度。

        参数:
        - X: 输入数据数组，形状为 (N, d_1, ..., d_k)
        - y: 标签数组，形状为 (N,)。y[i] 给出 X[i] 的标签。

        返回:
        如果 y 是 None，则执行模型的测试时前向传播，并返回:
        - scores: 形状为 (N, C) 的分类分数数组，其中 scores[i, c] 是 X[i] 和类别 c 的分类分数。

        如果 y 不是 None，则执行训练时的前向和反向传播，并返回一个元组：
        - loss: 一个标量值，表示损失
        - grads: 一个字典，与 self.params 有相同的键，将参数名称映射到损失相对于这些参数的梯度。
        """
        scores = None
        ############################################################################
        # TODO: 实现两层网络的前向传播，计算 X 的分类分数并将它们存储在 scores 变量中。              #
        ############################################################################
        # a1_out, a1_cache = affine_forward(X, self.params['W1'], self.params['b1'])
        # r1_out, r1_cache = relu_forward(a1_out)
        # 使用 affine_relu_forward 计算仿射-ReLU层的输出和缓存
        ar1_out, ar1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        # 使用 affine_forward 计算仿射层的输出和缓存
        a2_out, a2_cache = affine_forward(ar1_out, self.params['W2'], self.params['b2'])
        scores = a2_out

        # 如果 y 是 None，则处于测试模式，直接返回 scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: 实现两层网络的反向传播。将损失存储在 loss 变量中，梯度存储在 grads 字典中。        #
        # 使用 softmax 计算数据损失，确保 grads[k] 保存了 self.params[k] 的梯度。不要忘记添加 L2 正则化！#
        #                                                                          #
        # 注意：为了确保您的实现与我们的一致，并通过自动化测试，确保 L2 正则化包含一个因子 0.5，以简化梯度表达式。 #
        ############################################################################
        # 计算 softmax 损失和其相对于 scores 的梯度 dscores
        loss, dscores = softmax_loss(scores, y)
        # 添加 L2 正则化到损失
        loss = loss + 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1']) + 0.5 * self.reg * np.sum(
            self.params['W2'] * self.params['W2'])
        # 反向传播到第二层的梯度
        dx2, dw2, db2 = affine_backward(dscores, a2_cache)
        grads['W2'] = dw2 + self.reg * self.params['W2']
        grads['b2'] = db2
        # dx2_relu = relu_backward(dx2, r1_cache)
        # dx1, dw1, db1 = affine_backward(dx2_relu, a1_cache)
        # 反向传播到第一层的梯度
        dx1, dw1, db1 = affine_relu_backward(dx2, ar1_cache)
        grads['W1'] = dw1 + self.reg * self.params['W1']
        grads['b1'] = db1

        return loss, grads


class FullyConnectedNet(object):
    """
  一个全连接神经网络，具有任意数量的隐藏层，使用 ReLU 非线性激活函数和 softmax 损失函数。
  该网络还可以选择性地实现 dropout 和 batch normalization。对于具有 L 层的网络，其架构为：

  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

  其中 batch normalization 和 dropout 是可选的，{...} 部分重复 L - 1 次。

  与上面的 TwoLayerNet 类似，可学习的参数存储在 self.params 字典中，并将通过 Solver 类进行学习。
  """


    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
    初始化一个新的 FullyConnectedNet。

    参数:
    - hidden_dims: 一个整数列表，表示每个隐藏层的大小。
    - input_dim: 一个整数，表示输入的大小。
    - num_classes: 一个整数，表示分类的类别数。
    - dropout: 介于 0 和 1 之间的标量，表示 dropout 的强度。如果 dropout=0，则网络不使用 dropout。
    - use_batchnorm: 布尔值，是否使用 batch normalization。
    - reg: 一个标量，表示 L2 正则化强度。
    - weight_scale: 一个标量，表示随机初始化权重的标准差。
    - dtype: 一个 numpy 数据类型对象；所有计算将使用此数据类型进行。float32 较快但精度较低，因此数值梯度检查时建议使用 float64。
    - seed: 若非 None，则传递此随机种子给 dropout 层，以便 dropout 层具有确定性，从而可以对模型进行梯度检查。
    """
        self.use_batchnorm = use_batchnorm  # 是否使用批量归一化
        self.use_dropout = dropout > 0  # 是否使用 dropout
        self.reg = reg  # L2 正则化强度
        self.num_layers = 1 + len(hidden_dims)  # 总层数，包括输出层
        self.dtype = dtype  # 数据类型
        self.params = {}  # 存储可学习参数的字典

        ############################################################################
        # TODO: 初始化网络的参数，所有值存储在 self.params 字典中。第一层的权重和偏置使用 W1 和 b1，  #
        # 第二层使用 W2 和 b2，以此类推。权重应从标准差等于 weight_scale 的正态分布中初始化，偏置   #
        # 应初始化为零。                                                              #
        #                                                                          #
        # 如果使用 batch normalization，应存储每层的缩放和偏移参数，第一层使用 gamma1 和 beta1，     #
        # 第二层使用 gamma2 和 beta2，以此类推。缩放参数应初始化为 1，偏移参数初始化为 0。            #
        ############################################################################
        # 初始化每一层的权重和偏置

        layer_input_dim = input_dim  # 输入层维度
        for i, hd in enumerate(hidden_dims):  # 遍历每个隐藏层的维度
            self.params['W%d' % (i + 1)] = weight_scale * np.random.randn(layer_input_dim, hd)  # 权重初始化
            self.params['b%d' % (i + 1)] = weight_scale * np.zeros(hd)  # 偏置初始化
            # 如果使用 batch normalization，则初始化 gamma 和 beta 参数
            if self.use_batchnorm:
                self.params['gamma%d' % (i + 1)] = np.ones(hd)  # 缩放参数初始化为1
                self.params['beta%d' % (i + 1)] = np.zeros(hd)  # 偏移参数初始化为0
            layer_input_dim = hd  # 更新输入层维度
        # 初始化输出层的权重和偏置
        self.params['W%d' % (self.num_layers)] = weight_scale * np.random.randn(layer_input_dim, num_classes)
        self.params['b%d' % (self.num_layers)] = weight_scale * np.zeros(num_classes)

        # 当使用 dropout 时，我们需要将一个 dropout_param 字典传递给每个 dropout 层，以便该层知道 dropout 概率和模式（训练/测试）。
        # 你可以将相同的 dropout_param 传递给每个 dropout 层。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}  # 设置 dropout 参数
            if seed is not None:
                self.dropout_param['seed'] = seed  # 设置随机种子

        # 使用批量归一化时，需要跟踪运行均值和方差
        # 因此需要将一个特殊的 bn_param 对象传递给每个批量归一化层。
        # 应将 self.bn_params[0] 传递给第一个批量归一化层的前向传播，
        # 将 self.bn_params[1] 传递给第二个批量归一化层的前向传播，依此类推。
        self.bn_params = []  # 存储批量归一化参数
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]  # 初始化批量归一化参数

        # # 将所有参数转换为正确的数据类型
        # for k, v in self.params.iteritems():
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)  # 转换数据类型

    def loss(self, X, y=None):
        """
        计算全连接网络的损失和梯度。
        输入/输出：与上面的 TwoLayerNet 类似。
        """
        X = X.astype(self.dtype)  # 将输入转换为指定的数据类型
        mode = 'test' if y is None else 'train'  # 确定模式（训练或测试）

        # 为批量归一化参数和 dropout 参数设置训练/测试模式，因为它们在训练和测试期间的行为不同。
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode  # 设置每个批量归一化层的模式

        scores = None  # 初始化类别得分

        ############################################################################
        # TODO: 实现全连接网络的前向传播，计算输入 X 的类别得分，并将其存储在 scores 变量中。 #
        #                                                                          #
        # 当使用 dropout 时，需要将 self.dropout_param 传递给每个 dropout 前向传播。        #
        #                                                                          #
        # 当使用批量归一化时，需要将 self.bn_params[0] 传递给第一个批量归一化层的前向传播， #
        # 将 self.bn_params[1] 传递给第二个批量归一化层的前向传播，依此类推。     #
        ############################################################################
        layer_input = X  # 初始化输入层
        ar_cache = {}  # 存储激活和缓存
        dp_cache = {}  # 存储dropout 缓存

        # 遍历每个隐藏层
        for lay in range(self.num_layers - 1):
            if self.use_batchnorm:  # 如果使用批量归一化
                layer_input, ar_cache[lay] = affine_bn_relu_forward(layer_input,
                                                                    self.params['W%d' % (lay + 1)],
                                                                    self.params['b%d' % (lay + 1)],
                                                                    self.params['gamma%d' % (lay + 1)],
                                                                    self.params['beta%d' % (lay + 1)],
                                                                    self.bn_params[lay]) # 前向传播
            else:
                layer_input, ar_cache[lay] = affine_relu_forward(layer_input, self.params['W%d' % (lay + 1)],
                                                                 self.params['b%d' % (lay + 1)]) # 前向传播

            if self.use_dropout:
                layer_input, dp_cache[lay] = dropout_forward(layer_input, self.dropout_param) # 前向传播

        # 处理输出层
        ar_out, ar_cache[self.num_layers] = affine_forward(layer_input, self.params['W%d' % (self.num_layers)],
                                                           self.params['b%d' % (self.num_layers)])
        scores = ar_out  # 保存得分

        # 如果是测试模式则提前返回
        if mode == 'test':
            return scores
        loss, grads = 0.0, {}
        loss, grads = 0.0, {}  # 初始化损失和梯度
        ############################################################################
        # TODO: 实现全连接网络的反向传播。将损失存储在 loss 变量中，梯度存储在 grads 字典中。 #
        # 使用 softmax 计算数据损失，并确保 grads[k] 保存 self.params[k] 的梯度。     #
        # 不要忘记添加 L2 正则化！                                               #
        #                                                                          #
        # 当使用批量归一化时，不需要对缩放和偏移参数进行正则化。                  #
        #                                                                          #
        # 注意：为了确保您的实现与我们的实现一致，并通过自动测试，请确保 L2 正则化包括  #
        # 一个 0.5 的因子以简化梯度表达式。                                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)  # 使用 softmax 计算损失和梯度
        dhout = dscores  # 保存从 softmax 得到的梯度

        # 添加 L2 正则化损失
        loss = loss + 0.5 * self.reg * np.sum(
            self.params['W%d' % (self.num_layers)] * self.params['W%d' % (self.num_layers)])

        # 反向传播输出层
        dx, dw, db = affine_backward(dhout, ar_cache[self.num_layers])
        grads['W%d' % (self.num_layers)] = dw + self.reg * self.params['W%d' % (self.num_layers)] # 更新权重梯度
        grads['b%d' % (self.num_layers)] = db  # 更新偏置梯度
        dhout = dx  # 更新传递的梯度

        # 反向传播所有隐蔽层
        for idx in range(self.num_layers - 1):
            lay = self.num_layers - 1 - idx - 1  # 计算当前层索引
            # 添加每层的 L2 正则化损失
            loss = loss + 0.5 * self.reg * np.sum(self.params['W%d' % (lay + 1)] * self.params['W%d' % (lay + 1)])
            if self.use_dropout:  # 使用dropout
                dhout = dropout_backward(dhout, dp_cache[lay])  # 反向传播 dropout
            if self.use_batchnorm:  # 使用批量归一化
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhout, ar_cache[lay])  # 反向传播
            else:
                dx, dw, db = affine_relu_backward(dhout, ar_cache[lay])  # 反向传播
            # 更新梯度
            grads['W%d' % (lay + 1)] = dw + self.reg * self.params['W%d' % (lay + 1)]
            grads['b%d' % (lay + 1)] = db
            if self.use_batchnorm:  # 如果使用批量归一化
                grads['gamma%d' % (lay + 1)] = dgamma  # 更新 gamma 梯度
                grads['beta%d' % (lay + 1)] = dbeta  # 更新 beta 梯度
            dhout = dx  # 更新传递的梯度

        return loss, grads  # 返回损失和梯度