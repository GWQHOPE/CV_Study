import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
  """
    给定图像的像素数据和多个特征函数，应用所有特征函数到所有图像，
    将每个图像的特征向量连接起来，并将所有图像的特征存储在一个矩阵中。

    输入:
    - imgs: N x H x W x C 数组，表示 N 张图像的像素数据。
    - feature_fns: 特征函数列表，第 i 个特征函数应接受一个 H x W x D 数组作为输入
      并返回一个长度为 F_i 的一维数组。
    - verbose: 布尔值；如果为真，则打印进度。

    返回:
    一个形状为 (N, F_1 + ... + F_k) 的数组，其中每列是单个图像的所有特征的连接。
  """
  num_images = imgs.shape[0]
  if num_images == 0:
    return np.array([])

  # 使用第一张图像来确定特征维度
  feature_dims = []
  first_image_features = []
  for feature_fn in feature_fns:
    feats = feature_fn(imgs[0].squeeze())
    assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
    feature_dims.append(feats.size)
    first_image_features.append(feats)

  # 现在知道特征的维度，可以分配一个大的数组来存储所有特征
  total_feature_dim = sum(feature_dims)
  imgs_features = np.zeros((num_images, total_feature_dim))
  imgs_features[0] = np.hstack(first_image_features).T

  # 为其余图像提取特征
  for i in range(1, num_images):
    idx = 0
    for feature_fn, feature_dim in zip(feature_fns, feature_dims):
      next_idx = idx + feature_dim
      imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
      idx = next_idx
    if verbose and i % 1000 == 0:
      print('Done extracting features for %d / %d images' % (i, num_images))

  return imgs_features


def rgb2gray(rgb):
  """将 RGB 图像转换为灰度图像

    参数:
      rgb : RGB 图像

    返回:
      gray : 灰度图像
  """
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def hog_feature(im):
  """计算图像的梯度方向直方图 (HOG) 特征

       修改自 skimage.feature.hog
       http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

     参考文献:
       人体检测的梯度方向直方图
       Navneet Dalal 和 Bill Triggs, CVPR 2005

    参数:
      im : 输入的灰度或 RGB 图像

    返回:
      feat: 梯度方向直方图 (HOG) 特征
  """

  # 如果需要，转换 RGB 为灰度图像
  if im.ndim == 3:
    image = rgb2gray(im)
  else:
    image = np.at_least_2d(im)

  sx, sy = image.shape # image size
  orientations = 9 # number of gradient bins
  cx, cy = (8, 8) # pixels per cell

  gx = np.zeros(image.shape)
  gy = np.zeros(image.shape)
  gx[:, :-1] = np.diff(image, n=1, axis=1) # compute gradient on x-direction
  gy[:-1, :] = np.diff(image, n=1, axis=0) # compute gradient on y-direction
  grad_mag = np.sqrt(gx ** 2 + gy ** 2) # gradient magnitude
  grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90 # gradient orientation

  n_cellsx = int(np.floor(sx / cx))  # number of cells in x
  n_cellsy = int(np.floor(sy / cy))  # number of cells in y
  # compute orientations integral images
  orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
  for i in range(orientations):
    # create new integral image for this orientation
    # isolate orientations in this range
    temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                        grad_ori, 0)
    temp_ori = np.where(grad_ori >= 180 / orientations * i,
                        temp_ori, 0)
    # select magnitudes for those orientations
    cond2 = temp_ori > 0
    temp_mag = np.where(cond2, grad_mag, 0)
    # orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[cx/2::cx, cy/2::cy].T
    orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[int(cx / 2)::cx, int(cy / 2)::cy].T

  return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
  """
    计算图像的色彩直方图，使用色相（Hue）。

    输入:
    - im: H x W x C 数组，表示 RGB 图像的像素数据。
    - nbin: 直方图的箱数（默认: 10）
    - xmin: 最小像素值（默认: 0）
    - xmax: 最大像素值（默认: 255）
    - normalized: 是否归一化直方图（默认: True）

    返回:
      长度为 nbin 的 1D 向量，表示输入图像色相的颜色直方图。
  """
  ndim = im.ndim
  bins = np.linspace(xmin, xmax, nbin+1)
  hsv = matplotlib.colors.rgb_to_hsv(im/xmax) * xmax
  imhist, bin_edges = np.histogram(hsv[:,:,0], bins=bins, density=normalized)
  imhist = imhist * np.diff(bin_edges)

  # 返回直方图
  return imhist


pass
