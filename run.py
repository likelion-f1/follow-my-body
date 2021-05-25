# -*- coding: utf-8 -*-
import configparser
import cv2
# pycharm 에서 자동완성 되게 하기 위해
# try:
#     from cv2 import cv2
# except ImportError:
#     pass
import timeit
import numpy as np
from tensorflow.python.client import device_lib

import deeplab
import action_classifier
from VideoProperties import VideoProperties


dest_mask = ''
answer_outline = ''
answer = 'downdog'
labels = ['goddess', 'spiderman', 'tree', 'superman', 'warrior']
index = 0


def run_with_img(resized_frame):

    if resized_frame is None or resized_frame is False:
        return
    # print('frame count', video_properties.counter)
    resized_frame = cv2.flip(resized_frame, 1)

    # =========== segmentation 추출!
    # result = bodypix_model.predict_single(resized_frame)
    # my_mask = result.get_mask(threshold=0.7)

    start_time = timeit.default_timer()
    my_mask = deeplab.get_deeplab_mask(resized_frame, visualization=False)
    my_mask = cv2.resize(np.array(my_mask, dtype=np.uint8),
                         dsize=(resized_frame.shape[1], resized_frame.shape[0]),
                         interpolation=cv2.INTER_LINEAR)

    terminate_time = timeit.default_timer()  # 종료 시간 체크
    print("segmentation: %.3f초 " % (terminate_time - start_time))

    # my_mask[my_mask==1] = 255
    my_mask[my_mask != 15] = 0
    my_mask[my_mask == 15] = 255

    # =========== 유사도 계산
    # hash = get_similarity(resized_frame, answer_outline, type='hash')
    hamming = get_similarity(my_mask, dest_mask, type='hamming')

    # =========== CNN 기반 action 유사도
    # label, prob = action_classifier.get_action_class(my_mask)

    # =========== outline, 유사도 결과 그리기
    if hamming > 0.9:
        show_outline(resized_frame, get_outline(my_mask), color=(12, 12, 250), thick=5)
        set_new_answer(resized_frame.shape[0], resized_frame.shape[1])
    else:
        show_outline(resized_frame, get_outline(my_mask), color=(12, 222, 250))
    show_outline(resized_frame, answer_outline, color=(53, 103, 2), thick=3)
    # show_similarity(resized_frame, '%s, %.2f' % (label, prob), cnt=1, color=(115, 50, 168))
    show_similarity(resized_frame, 'ham: %.1f %%' % (hamming*100), cnt=2, color=(15, 166, 247))

    # cv2.imshow('frame', resized_frame)

    # ESC 키 클릭 시
    # if cv2.waitKey(1) == 27:
    #     # cv2.imwrite('./imgs/cap.jpg', resized_frame)
    #     stop = True
    #     cv2.destroyAllWindows()
    #     break
    return resized_frame


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

    # 샘플 정답지 set to g
    set_new_answer(height, width)

    while resized_frame is not None and resized_frame is not False:
        print('frame count', video_properties.counter)
        resized_frame = cv2.flip(resized_frame, 1)

        # =========== segmentation 추출!
        # =========== bodypix model
        # result = bodypix_model.predict_single(resized_frame)
        # my_mask = result.get_mask(threshold=0.7)

        # =========== deeplab model
        start_time = timeit.default_timer()
        my_mask = deeplab.get_deeplab_mask(resized_frame, visualization=False)
        my_mask = cv2.resize(np.array(my_mask, dtype=np.uint8),
                             dsize=(resized_frame.shape[1], resized_frame.shape[0]),
                             interpolation=cv2.INTER_LINEAR)

        terminate_time = timeit.default_timer()  # 종료 시간 체크
        print("segmentation: %.3f초 " % (terminate_time - start_time))

        my_mask[my_mask != 15] = 0
        my_mask[my_mask == 15] = 255

        # =========== 유사도 계산
        hash = get_similarity(resized_frame, answer_outline, type='hash')
        hamming = get_similarity(my_mask, dest_mask, type='hamming')

        # =========== CNN 기반 action 유사도
        label, prob = action_classifier.get_action_class(my_mask)

        # =========== outline, 유사도 결과 그리기
        if hamming > 0.9:
            show_outline(resized_frame, get_outline(my_mask), color=(12, 12, 250), thick=5)
            set_new_answer(resized_frame.shape[0], resized_frame.shape[1])
        else:
            show_outline(resized_frame, get_outline(my_mask), color=(12, 222, 250))
        show_outline(resized_frame, answer_outline, color=(53, 103, 2), thick=3)
        show_similarity(resized_frame, '%s, %.2f' % (label, prob), cnt=1, color=(115, 50, 168))
        show_similarity(resized_frame, 'ham: %.1f %%' % (hamming*100), cnt=2, color=(15, 166, 247))
        cv2.imshow('frame', resized_frame)

        # time.sleep(1/30)
        resized_frame = video_properties.play_video(video_capture)

        key = cv2.waitKey(1)
        # SPACE 키 클릭 시
        if key == 32:
            set_new_answer(height, width)
            print('자세 change!')
            continue

        # ESC 키 클릭 시
        if key == 27:
            # cv2.imwrite('./imgs/cap.jpg', resized_frame)
            stop = True
            cv2.destroyAllWindows()
            break

    return stop


def set_new_answer(height, width):
    global dest_mask
    global answer_outline
    global answer
    global index

    # get random new answer
    # new_answer = labels[np.random.randint(0, len(labels))]
    index = (index+1) % len(labels)
    answer = labels[index]
    print('---------------' + answer + '---------------')

    # if answer == new_answer:
    #     new_answer = labels[np.random.randint(0, len(labels))]
    # answer = new_answer

    # 샘플 정답지
    sample_mask = cv2.imread('./imgs/' + answer + '.jpg', cv2.IMREAD_GRAYSCALE)

    # if answer == 'superman':
    #     sample_mask = np.array(sample_mask)
    #     sample_mask = cv2.resize(sample_mask, dsize=(260, 340))
    #     sample_mask = sample_mask[4:-4, 4:-4]
    #     sample_mask[sample_mask!=0] = 255
    #     cv2.imwrite('./imgs/' + answer + '.jpg', sample_mask)

    # return

    # sample_mask = sample_mask[:-20]
    div = 1.1
    add = -5
    if answer == 'superman':
        div = 0.8
    # if answer == 'goddess':
        sample_mask = sample_mask[:-30]
    if answer == 'squat':
        div = 0.9
    if answer == 'plank':
        sample_mask = sample_mask[:-45]
        div = 1.3
    if answer == 'downdog':
        sample_mask = sample_mask[:-20]
        div = 1.5
    if answer == 'warrior':
        sample_mask = sample_mask[:-50]
    if answer == 'tree':
        add = -20

    s_width, s_height = sample_mask.shape[1], sample_mask.shape[0]
    s_width, s_height = int(s_width / div), int(s_height / div)
    sample_mask = cv2.resize(sample_mask, dsize=(s_width, s_height), interpolation=cv2.INTER_AREA)
    dest_mask = np.zeros((height, width), dtype=np.uint8)
    try:
        dest_mask[ add - s_height:add, int(width / 2 - s_width / 2):int(width / 2 - s_width / 2) + s_width] = sample_mask
        answer_outline = get_outline(dest_mask)
    except Exception as e:
        set_new_answer(height, width)
        # pass


# mask 에서 테두리만 추출 (그리기 용)
def get_outline(mask):
    mask_edge = cv2.Canny(mask.astype(np.uint8), 50, 200)
    mask_edge_p = np.where(mask_edge == 255)
    mask_edge_p = [(y, x) for x, y in zip(mask_edge_p[0], mask_edge_p[1])]
    return mask_edge_p


# image 분석 결과와 정답 mask 간의 유사도
def get_similarity(src_mask, dest_mask, type='hash'):
    # 생김새 유사도
    if type == 'hash':
        sim = ''
    # 픽셀 ROI
    elif type == 'hamming':
        print((src_mask == 255).sum(), (dest_mask == 255).sum())
        kyo = ((src_mask == 255) & (dest_mask == 255)).sum()
        hap = ((src_mask == 255) | (dest_mask == 255)).sum()
        print('kyo, han', kyo, hap)
        result = (kyo / hap) * 1.64
        if result > 1: result = 0.99
        return result
    return sim


def show_similarity(image, sim, color=(150, 150, 10), cnt=1):
    cv2.putText(image, sim, (image.shape[1] - 250, cnt * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)


def show_outline(image, mask_edge_p, color=(200, 255, 0), thick=3):
    for x, y in mask_edge_p:
        cv2.circle(image, (x, y), 0, color, thick)


def load_bodypix_model():
    from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
    print('bodypix 모델 로드 중...')
    bodypix_model = load_model(download_model(
        BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
    ))
    print('bodypix 모델 로드 완료!')


if __name__ == '__main__':

    print('======================================')
    # print('DEVICE LIST:\n', device_lib.list_local_devices())
    print('======================================')

    ini = configparser.ConfigParser()
    ini.read('config.ini', encoding='utf-8')
    run(VideoProperties(ini))

    # while True:
    #     stop = run(VideoProperties(ini))
    #     if stop:
    #         break

    # for i in range(1600):
    #     cap = cv2.VideoCapture(i)
    #     success, image = cap.read()
    #     if success:
    #         print(i)
    #     cap.release()
