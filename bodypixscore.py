import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import numpy as np
import sys
from PIL import Image
import imagehash

bodypix_model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
))

poseImage = '이미지이름'
myImage = '이미지이름'

# array 형식의 mask를 구한다
def img_to_mask(img_name, thres=0.5):
    image = tf.keras.preprocessing.image.load_img(img_name)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    result = bodypix_model.predict_single(image_array)
    mask = result.get_mask(threshold=thres)
    tf.keras.preprocessing.image.save_img('mask_' + img_name , mask)
    h, w = image_array.shape[:2]
    return mask.numpy().reshape(h, w)
    
marr_pose = img_to_mask(poseImage)
marr_mine = img_to_mask(myImage)

# 픽셀 간해밍거리 이용한 점수
def hamming_score(arr1, arr2):
    arr1 = arr1.reshape(1, -1)
    arr2 = arr2.reshape(1, -1)
    distance = (arr1 != arr2).sum()
    return (h * w - distance) / (h * w)

print(hamming_score(marr_pose, marr_mine))

def hash_score(hash_size=8):
    hash1 = imagehash.dhash(Image.open('mask_' + poseImage), hash_size)
    hash2 = imagehash.dhash(Image.open('mask_' + myImage), hash_size)
    return (hash_size ** 2 - (hash1 - hash2)) / hash_size ** 2

print(hash_score())