from math import sqrt, ceil
import numpy as np

def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    将4D张量的图像数据重塑为网格，便于可视化。

    输入:
    - Xs: 形状为 (N, H, W, C) 的数据
    - ubound: 输出网格的值将缩放到范围 [0, ubound]
    - padding: 网格元素之间的空白像素数量
    """
    (N, H, W, C) = Xs.shape  # 获取输入数据的维度
    grid_size = int(ceil(sqrt(N)))  # 计算网格大小，基于图像数量N
    grid_height = H * grid_size + padding * (grid_size - 1)  # 网格的高度
    grid_width = W * grid_size + padding * (grid_size - 1)  # 网格的宽度
    grid = np.zeros((grid_height, grid_width, C))  # 初始化网格
    next_idx = 0  # 跟踪当前图像索引
    y0, y1 = 0, H  # y轴起始和结束位置
    for y in range(grid_size):
        x0, x1 = 0, W  # x轴起始和结束位置
        for x in range(grid_size):
            if next_idx < N:  # 如果还有图像可用
                img = Xs[next_idx]  # 获取当前图像
                low, high = np.min(img), np.max(img)  # 找到图像的最小和最大值
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)  # 归一化并放入网格
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid  # 返回构建的网格

def vis_grid(Xs):
    """ 可视化图像网格 """
    (N, H, W, C) = Xs.shape  # 获取输入数据的维度
    A = int(ceil(sqrt(N)))  # 计算网格大小
    G = np.ones((A*H+A, A*W+A, C), Xs.dtype)  # 初始化网格
    G *= np.min(Xs)  # 用最小值填充网格
    n = 0  # 图像索引
    for y in range(A):
        for x in range(A):
            if n < N:  # 如果还有图像可用
                G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = Xs[n, :, :, :]  # 将图像放入网格
                n += 1  # 更新索引
    # 归一化到 [0, 1] 范围
    maxg = G.max()  # 网格的最大值
    ming = G.min()  # 网格的最小值
    G = (G - ming) / (maxg - ming)  # 归一化处理
    return G  # 返回处理后的网格

def vis_nn(rows):
    """ 可视化一组图像数组 """
    N = len(rows)
    D = len(rows[0])
    H, W, C = rows[0][0].shape  # 获取图像的高度、宽度和通道数
    Xs = rows[0][0]  # 获取第一张图像用于初始化
    G = np.ones((N*H+N, D*W+D, C), Xs.dtype)  # 初始化网格
    for y in range(N):
        for x in range(D):
            G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = rows[y][x]  # 将图像放入网格
    # 归一化到 [0, 1] 范围
    maxg = G.max()  # 网格的最大值
    ming = G.min()  # 网格的最小值
    G = (G - ming) / (maxg - ming)  # 归一化处理
    return G  # 返回处理后的网格