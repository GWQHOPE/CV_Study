import pickle as pickle
import numpy as np
import os
# from scipy.misc import imread
from imageio import imread

def load_CIFAR_batch(filename):
  """ 加载单个CIFAR批次 """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f, encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    # 将数据从 (10000, 3072) 形状重塑为 (10000, 3, 32, 32)，并转换为浮点数类型
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """加载整个CIFAR-10数据集"""
  xs = []
  ys = []
  # 遍历5个数据批次
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  # 合并训练数据和标签
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y  # 删除不再需要的变量
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))  # 加载测试批次
  return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    从磁盘加载CIFAR-10数据集，并进行预处理，以便为分类器做好准备。
    这些步骤与我们为SVM使用的相同，但浓缩为一个函数。
    """
    # 加载原始CIFAR-10数据
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # 子采样数据
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask] # 验证集
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]  # 训练集
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]  # 测试集
    y_test = y_test[mask]

    # 数据标准化：减去均值图像
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # 转置，使得通道数在第一维
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # 将数据打包成字典
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }
    

def load_tiny_imagenet(path, dtype=np.float32):
  """
  加载TinyImageNet。每个TinyImageNet-100-A、TinyImageNet-100-B和
    TinyImageNet-200具有相同的目录结构，因此可以用于加载其中的任何一个。

    输入：
    - path: 字符串，给出要加载的目录路径。
    - dtype: 用于加载数据的numpy数据类型。

    返回：一个元组
    - class_names: 一个列表，其中class_names[i]是一个字符串列表，给出
      加载的数据集中类i的WordNet名称。
    - X_train: (N_tr, 3, 64, 64) 训练图像数组
    - y_train: (N_tr,) 训练标签数组
    - X_val: (N_val, 3, 64, 64) 验证图像数组
    - y_val: (N_val,) 验证标签数组
    - X_test: (N_test, 3, 64, 64) 测试图像数组。
    - y_test: (N_test,) 测试标签数组；如果测试标签不可用
      （例如在学生代码中），则y_test将为None。
  """
  # 首先加载wnids
  with open(os.path.join(path, 'wnids.txt'), 'r') as f:
    wnids = [x.strip() for x in f]

  # 将wnids映射到整数标签
  wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

  # 使用words.txt获取每个类的名称
  with open(os.path.join(path, 'words.txt'), 'r') as f:
    wnid_to_words = dict(line.split('\t') for line in f)
    for wnid, words in wnid_to_words.items():
      wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
  class_names = [wnid_to_words[wnid] for wnid in wnids]

  # 接下来加载训练数据
  X_train = []
  y_train = []
  for i, wnid in enumerate(wnids):
    if (i + 1) % 20 == 0:
      print('loading training data for synset %d / %d' % (i + 1, len(wnids)))
    # 确定我们需要打开的文件名
    boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
    with open(boxes_file, 'r') as f:
      filenames = [x.split('\t')[0] for x in f]
    num_images = len(filenames)

    # 创建用于存储图像的数组
    X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
    y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
    for j, img_file in enumerate(filenames):
      img_file = os.path.join(path, 'train', wnid, 'images', img_file)
      img = imread(img_file)
      if img.ndim == 2:
        # 如果是灰度图像
        img.shape = (64, 64, 1)
      X_train_block[j] = img.transpose(2, 0, 1) # 转置为 (3, 64, 64)
    X_train.append(X_train_block)
    y_train.append(y_train_block)
      
  # 需要合并所有训练数据
  X_train = np.concatenate(X_train, axis=0)
  y_train = np.concatenate(y_train, axis=0)
  
  # 接下来加载验证数据
  with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
    img_files = []
    val_wnids = []
    for line in f:
      img_file, wnid = line.split('\t')[:2]
      img_files.append(img_file)
      val_wnids.append(wnid)
    num_val = len(img_files)
    y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
    X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
      img_file = os.path.join(path, 'val', 'images', img_file)
      img = imread(img_file)
      if img.ndim == 2:
        img.shape = (64, 64, 1)
      X_val[i] = img.transpose(2, 0, 1)

  # 接下来加载测试图像
  # 学生没有测试标签，所以我们需要遍历文件夹中的文件。
  img_files = os.listdir(os.path.join(path, 'test', 'images'))  # 获取测试图像文件列表
  X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)  # 初始化测试数据数组
  for i, img_file in enumerate(img_files):
    img_file = os.path.join(path, 'test', 'images', img_file)  # 构造完整的图像路径
    img = imread(img_file)  # 读取图像
    if img.ndim == 2:  # 如果是灰度图像
      img.shape = (64, 64, 1)  # 将形状调整为 (64, 64, 1)
    X_test[i] = img.transpose(2, 0, 1)  # 转置图像为 (3, 64, 64) 格式

  y_test = None # 初始化测试标签为 None
  y_test_file = os.path.join(path, 'test', 'test_annotations.txt') # 测试标签文件路径
  if os.path.isfile(y_test_file):  # 检查测试标签文件是否存在
    with open(y_test_file, 'r') as f:
      img_file_to_wnid = {} # 初始化图像文件到 WordNet ID 的映射字典
      for line in f:
        line = line.split('\t')  # 按制表符分割行
        img_file_to_wnid[line[0]] = line[1]  # 将文件名和对应的 WordNet ID 存入字典
    # 根据图像文件获取标签并存储到 y_test 数组
    y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
    y_test = np.array(y_test)  # 将 y_test 转换为 NumPy 数组

  # 返回类名、训练集、验证集和测试集数据及标签
  return class_names, X_train, y_train, X_val, y_val, X_test, y_test


def load_models(models_dir):
  """
  从磁盘加载保存的模型。此函数将尝试反序列化目录中的所有文件；
    任何在反序列化时出现错误的文件（如 README.txt）将被跳过。

    输入：
    - models_dir: 字符串，给出包含模型文件的目录路径。
      每个模型文件是一个带有 'model' 字段的序列化字典。

    返回：
    一个将模型文件名映射到模型的字典。
  """
  models = {}  # 初始化模型字典
  for model_file in os.listdir(models_dir):  # 遍历模型目录中的所有文件
    with open(os.path.join(models_dir, model_file), 'rb') as f:
      try: # 如果反序列化失败 跳过文件
        models[model_file] = pickle.load(f)['model']
      except pickle.UnpicklingError:
        continue
  return models
