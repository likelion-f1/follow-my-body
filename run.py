# -*- coding: utf-8 -*-
import configparser
import collections
import time
# import cv2
# pycharm 에서 자동완성 되게 하기 위해
try:
    from cv2 import cv2
except ImportError:
    pass
import numpy as np
from PIL import Image
from VideoProperties import VideoProperties

import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths


def run(video_properties):
    stop = False

    # video_capture = cv2.VideoCapture(video_properties.video_path)
    video_capture = cv2.VideoCapture(0) # 카메라
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)
    video_properties.set_video_info(width, height, video_properties.resized_width)
    video_capture.set(1, video_properties.start_frame)
    resized_frame = video_properties.play_video(video_capture)

    # 정답지 그리기
    sample_mask = cv2.imread('./imgs/mask.jpg')
    sample_mask = np.array(sample_mask)

    sample_mask = cv2.resize(sample_mask, dsize=(sample_mask.shape[1]//3, sample_mask.shape[0]//3), interpolation=cv2.INTER_AREA)
    dest_mask = np.empty((height, width, 3))
    dest_mask.fill(0)
    s_height, s_width = sample_mask.shape[0], sample_mask.shape[1]
    dest_mask[-20 - s_height: -20, int(width/2 - s_width/2):int(width/2 - s_width/2) + s_width, :] = sample_mask
    # answer_outline = get_sample_answer_outline(mask2)
    answer_outline = get_outline(dest_mask)

    while resized_frame is not None and resized_frame is not False:
        print('frame count', video_properties.counter)
        resized_frame = cv2.flip(resized_frame, 1)

        # 유사도 계산
        result = bodypix_model.predict_single(resized_frame)
        my_mask = result.get_mask(threshold=0.7)
        my_mask = np.array(my_mask, dtype=np.uint8)
        my_mask[my_mask==1] = 255
        # cv2.imshow('mask', my_mask)

        hash = get_similarity(resized_frame, answer_outline, type='hash')
        hamming = get_similarity(resized_frame, answer_outline, type='hamming')

        show_outline(resized_frame, get_outline(my_mask), color=(12, 222, 250))
        # show_outline(resized_frame, answer_outline, color=(53, 107, 2), thick=5)
        show_similarity(resized_frame, 'hash:' + hash, cnt=1, color=(115, 50, 168))
        show_similarity(resized_frame, 'ham:' + hamming, cnt=2, color=(15, 166, 247))

        cv2.imshow('frame', resized_frame)

        time.sleep(1/30)
        resized_frame = video_properties.play_video(video_capture)

        if cv2.waitKey(1) == 27:
            cv2.imwrite('./imgs/cap.jpg', resized_frame)
            stop = True
            cv2.destroyAllWindows()
            break

    return stop


# FROM SAMPLE MASK
def get_sample_answer_outline(mask):
    mask_edge = cv2.Canny(mask.astype(np.uint8), 50, 200)
    mask_edge_p = np.where(mask_edge == 255)
    mask_edge_p = [(y, x) for x, y in zip(mask_edge_p[0], mask_edge_p[1])]
    return mask_edge_p


def get_outline(mask):
    mask_edge = cv2.Canny(mask.astype(np.uint8), 50, 200)
    mask_edge_p = np.where(mask_edge == 255)
    mask_edge_p = [(y, x) for x, y in zip(mask_edge_p[0], mask_edge_p[1])]
    return mask_edge_p


# image 분석 결과와 정답 mask 간의 유사도
def get_similarity(image, mask_edge_p, type='hash'):
    if type == 'hash':
        sim = ''
    elif type == 'hamming':
        sim = ''
    return sim


def show_similarity(image, sim, color=(150, 150, 10), cnt=1):
    cv2.putText(image, sim + ' %', (image.shape[1] - 200, cnt * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)


def show_outline(image, mask_edge_p, color=(200, 255, 0), thick=3):
    for x, y in mask_edge_p:
        cv2.circle(image, (x, y), 0, color, thick)


if __name__ == '__main__':
    ini = configparser.ConfigParser()
    ini.read('config.ini', encoding='utf-8')
    # run(VideoProperties(ini))
    print('bodypix 모델 로드 중...')
    bodypix_model = load_model(download_model(
        BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
    ))
    print('bodypix 모델 로드 완료!')

    while True:
        stop = run(VideoProperties(ini))
        if stop:
            break

