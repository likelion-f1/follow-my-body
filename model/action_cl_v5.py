# 구글코랩 이용 모델
# %tensorflow_version 1.x
from google.colab import drive
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import load_model
from tensorflow import keras
import os
import re
import glob
import cv2
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
drive.mount('/gdrive', force_remount=True)
# 반복되는 드라이브 경로 변수화
drive_path = '/gdrive/My Drive/temp/'
# 딥랩 모델 클래스


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.
        Args:
          image: A PIL.Image object, raw input image.
        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 0.5 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(
            target_size, Image.ANTIALIAS)
        # 이미지 리사이즈
        resized_image = resized_image.resize((500, 500))
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.
    Args:
      label: A 2D array with integer type, storing the segmentation label.
    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.
    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    colormap = create_pascal_label_colormap()
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
    return colormap[label]


def vis_segmentation(image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')
    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')
    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')
    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
        FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
# @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']
MODEL_NAME = 'xception_coco_voctrainval'
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}
_TARBALL_NAME = 'deeplab_model.tar.gz'
model_dir = tempfile.mkdtemp()
tf.gfile.MakeDirs(model_dir)
download_path = os.path.join(model_dir, _TARBALL_NAME)
print('downloading model, this might take a while...')
urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                           download_path)
print('download completed! loading DeepLab model...')
MODEL = DeepLabModel(download_path)
print('model loaded successfully!')
# Input: ndarray (cv2 image)


def get_deeplab_mask(img, RGB=False, visualization=False):
    if RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('running deeplab on image ...')
    resized_im, seg_map = MODEL.run(Image.fromarray(img))
    if visualization:
        # BGR 2 RGB
        if not RGB:
            resized_im = Image.merge("RGB", resized_im.split()[::-1])
        vis_segmentation(resized_im, seg_map)
    return seg_map


# 구글 드라이브 내 이미 경로 설정
downdog = [cv2.imread(
    f'/gdrive/My Drive/temp/TRAIN/downdog/image{i}.jpg') for i in range(1, 100)]
goddess = [cv2.imread(
    f'/gdrive/My Drive/temp/TRAIN/goddess/image{i}.jpg') for i in range(1, 100)]
plank = [cv2.imread(
    f'/gdrive/My Drive/temp/TRAIN/plank/image{i}.jpg') for i in range(1, 100)]
tree = [cv2.imread(
    f'/gdrive/My Drive/temp/TRAIN/tree/image{i}.jpg') for i in range(1, 100)]
warrior = [cv2.imread(
    f'/gdrive/My Drive/temp/TRAIN/warrior2/image{i}.jpg') for i in range(1, 100)]
files = [downdog, goddess, plank, tree, warrior]
num_classes = len(files)
X = []
Y = []
for index, cate in enumerate(files):
    label = [0 for i in range(num_classes)]
    label[index] = 1
    for j in cate:
        np_array = get_deeplab_mask(j, RGB=True, visualization=True)
        X.append(np_array)
        Y.append(label)
        print(label)
        print(Y)
X = np.array(X)
Y = np.array(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
X_train = X_train.reshape(-1, 500, 500, 1)
X_test = X_test.reshape(-1, 500, 500, 1)
# CNN 모델 수립
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 1)),
    layers.MaxPooling2D((3, 3), strides=(3, 3)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((3, 3), strides=(3, 3)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((3, 3), strides=(3, 3)),
    layers.Conv2D(100, (3, 3), activation='relu'),
    layers.MaxPooling2D((3, 3), strides=(3, 3)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(rate=0.3),
    layers.Dense(num_classes, activation='softmax')
])
# 모델 요약
model.summary()
# 모델 컴파일
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# 모델 학습
history = model.fit(X_train, Y_train, epochs=10)
# 모델 평가
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.evaluate(X_test, Y_test, verbose=0)
# 학습 곡선 그래프 확인
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
# plt.gca().set_ylim(0.85, 1.0)
# plt.gca().set_ylim(0, 0.5)
plt.show()
model.save(groups_folder_path + '/action_cl_model_v1')
# 라벨 예측 확인
for n in range(len(Y_test)):
    plt.imshow(X_test[n].reshape(384, 384),
               cmap='Greys', interpolation='nearest')
    tmp = "Label:" + str(np.argmax(Y_test[n])) + ", Prediction:" + str(
        model.predict_classes(X_test[n].reshape((1, 384, 384, 1))))
    plt.title(tmp)
    plt.show()
