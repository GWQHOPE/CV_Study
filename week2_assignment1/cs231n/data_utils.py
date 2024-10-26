import pickle as pickle
import numpy as np
import os
# from scipy.misc import imread
from imageio import imread

def load_CIFAR_batch(filename):
  """ 加载单个 CIFAR 数据批次 """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f,encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ 加载所有 CIFAR 数据 """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

def load_tiny_imagenet(path, dtype=np.float32):
  """
    加载 TinyImageNet。每个 TinyImageNet-100-A、TinyImageNet-100-B 和
    TinyImageNet-200 都具有相同的目录结构，因此可以用于加载任意数据集。

    输入:
    - path: 指定要加载的目录路径的字符串。
    - dtype: 加载数据使用的 numpy 数据类型。

    返回: 包含以下内容的元组
    - class_names: 一个列表，其中 class_names[i] 是一个字符串列表，给出
      加载数据集中类 i 的 WordNet 名称。
    - X_train: (N_tr, 3, 64, 64) 的训练图像数组
    - y_train: (N_tr,) 的训练标签数组
    - X_val: (N_val, 3, 64, 64) 的验证图像数组
    - y_val: (N_val,) 的验证标签数组
    - X_test: (N_test, 3, 64, 64) 的测试图像数组。
    - y_test: (N_test,) 的测试标签数组；如果测试标签不可用
      （例如在学生代码中），则 y_test 将为 None。
  """
  # 首先加载 wnids
  with open(os.path.join(path, 'wnids.txt'), 'r') as f:
    wnids = [x.strip() for x in f]

  # 将 wnids 映射到整数标签
  wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

  # 使用 words.txt 获取每个类的名称
  with open(os.path.join(path, 'words.txt'), 'r') as f:
    wnid_to_words = dict(line.split('\t') for line in f)
    for wnid, words in wnid_to_words.items():
      wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
  class_names = [wnid_to_words[wnid] for wnid in wnids]

  # Next load training data.
  X_train = []
  y_train = []
  for i, wnid in enumerate(wnids):
    if (i + 1) % 20 == 0:
      print('loading training data for synset %d / %d' % (i + 1, len(wnids)))
    # To figure out the filenames we need to open the boxes file
    boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
    with open(boxes_file, 'r') as f:
      filenames = [x.split('\t')[0] for x in f]
    num_images = len(filenames)
    
    X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
    y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
    for j, img_file in enumerate(filenames):
      img_file = os.path.join(path, 'train', wnid, 'images', img_file)
      img = imread(img_file)
      if img.ndim == 2:
        ## grayscale file
        img.shape = (64, 64, 1)
      X_train_block[j] = img.transpose(2, 0, 1)
    X_train.append(X_train_block)
    y_train.append(y_train_block)
      
  # We need to concatenate all training data
  X_train = np.concatenate(X_train, axis=0)
  y_train = np.concatenate(y_train, axis=0)
  
  # Next load validation data
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
  # 学生没有测试标签，因此需要遍历图像目录中的文件。
  img_files = os.listdir(os.path.join(path, 'test', 'images'))
  X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
  for i, img_file in enumerate(img_files):
    img_file = os.path.join(path, 'test', 'images', img_file)
    img = imread(img_file)
    if img.ndim == 2:
      img.shape = (64, 64, 1)
    X_test[i] = img.transpose(2, 0, 1)

  y_test = None
  y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
  if os.path.isfile(y_test_file):
    with open(y_test_file, 'r') as f:
      img_file_to_wnid = {}
      for line in f:
        line = line.split('\t')
        img_file_to_wnid[line[0]] = line[1]
    y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
    y_test = np.array(y_test)
  
  return class_names, X_train, y_train, X_val, y_val, X_test, y_test


def load_models(models_dir):
  """
    从磁盘加载保存的模型。将尝试解压缩目录中的所有文件；
    任何在解压缩时出错的文件（例如 README.txt）将被跳过。

    输入:
    - models_dir: 字符串，给出包含模型文件的目录路径。
      每个模型文件是一个包含 'model' 字段的 pickled 字典。

    返回:
    一个字典，将模型文件名映射到模型。
  """
  models = {}
  for model_file in os.listdir(models_dir):
    with open(os.path.join(models_dir, model_file), 'rb') as f:
      try:
        models[model_file] = pickle.load(f)['model']
      except pickle.UnpicklingError:
        continue
  return models
