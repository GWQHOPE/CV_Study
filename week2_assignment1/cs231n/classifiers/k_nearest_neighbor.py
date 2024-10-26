import numpy as np


class KNearestNeighbor(object):
    """ 一个使用L2距离的k近邻分类器 """

    def __init__(self):
        pass

    def train(self, X, y):
        """
   训练分类器。对于k近邻算法来说，就是记住训练数据。

     输入:
    - X: 形状为(num_train, D)的numpy数组，包含训练数据，由num_train个样本组成，每个样本的维度为D。
    - y: 形状为(N,)的numpy数组，包含训练标签，其中y[i]是X[i]的标签。
    """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
    使用该分类器预测测试数据的标签。

    输入:
    - X: 形状为(num_test, D)的numpy数组，包含测试数据，由num_test个样本组成，每个样本的维度为D。
    - k: 投票预测标签的最近邻的数量。
    - num_loops: 确定使用哪种实现来计算训练点和测试点之间的距离。

    返回
    -  y: 形状为(num_test,)的numpy数组，包含测试数据的预测标签，其中y[i]是测试点X[i]的预测标签。
    """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
    使用嵌套循环计算测试点X与训练点self.X_train之间的距离。

    输入:
    - X: 形状为(num_test, D)的numpy数组，包含测试数据。

    返回:
    - dists: 形状为(num_test, num_train)的numpy数组，
    其中dists[i, j]是第i个测试点与第j个训练点之间的欧几里得距离。
    """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # 计算第i个测试点与第j个训练点之间的l2距离，并将结果存储在dists[i, j]中        #
                # 不应该在维度上使用循环。                                               #
                #####################################################################
                # pass
                dists[i][j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
                #####################################################################
                #                       END OF YOUR CODE                            #
                #####################################################################
        return dists

    def compute_distances_one_loop(self, X):
        """
     使用单个循环计算测试点X与训练点self.X_train之间的距离。

    输入/输出: 同上面定义的方法compute_distances_two_loops
    """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # 计算第i个测试点与所有训练点之间的l2距离，并将结果存储在dists[i, :]中。         #
            #######################################################################
            # pass
            dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis=1))
            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################
        return dists

    def compute_distances_no_loops(self, X):
        """
    在不使用显式循环的情况下计算测试点X与训练点self.X_train之间的距离。

    输入/输出: 同上面定义的方法compute_distances_two_loops
    """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # 在没有任何显式循环的情况下，计算所有测试点与所有训练点之间的l2距离，并将结果存储在    #
        # dists中。                                                              #
        # 你应该仅使用基本的数组操作来实现这个函数，特别是，你不应该使用scipy中的函数。        #
        # 提示：尝试使用矩阵乘法和两个广播和来表述l2距离。                               #
        #########################################################################
        # pass
        dists = np.sqrt(-2 * np.dot(X, self.X_train.T) + np.sum(np.square(self.X_train), axis=1) + np.transpose(
            [np.sum(np.square(X), axis=1)]))
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists

    def predict_labels(self, dists, k=1):
        """
   给定测试点和训练点之间距离的矩阵，为每个测试点预测标签。

    输入:
    - dists: 形状为(num_test, num_train)的numpy数组，其中dists[i, j]
      给出第i个测试点与第j个训练点之间的距离。

    返回:
    - y: 形状为(num_test,)的numpy数组，包含测试数据的预测标签，
         其中y[i]是测试点X[i]的预测标签。
    """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # 一个长度为k的列表，存储与第i个测试点最近的k个邻居的标签。
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # 使用距离矩阵找到第i个测试点的k个最近邻，并使用self.y_train找到这些邻居的标签。将这  #
            # 些标签存储在closest_y中。                                                #
            # 提示：查找numpy.argsort函数。                                            #
            #########################################################################
            # pass
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            #########################################################################
            # TODO:                                                                 #
            # 现在你已经找到了k个最近邻的标签，你需要找到这些标签中出现最频繁的标签。             #
            # 将这个标签存储在y_pred[i]中。遇到平局时选择较小的标签。                        #
            #########################################################################
            # pass
            y_pred[i] = np.argmax(np.bincount(closest_y))
            #########################################################################
            #                           END OF YOUR CODE                            #
            #########################################################################

        return y_pred

