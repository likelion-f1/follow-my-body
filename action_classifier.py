import cv2
import numpy as np
from tensorflow.keras import models

model = models.load_model('./model/action_cl_model_v4')
# labels = ["batman", "ironman", "joker", 'spiderman', 'squat', 'superman', 'wonderwoman']
labels = ['downdog', 'goddess', 'plank', 'tree', 'warrior']


def get_action_class(img):
    img = cv2.resize(img, dsize=(500, 500))/255
    pred = model.predict(img.reshape((1, 500, 500, 1)))
    # print('pred', pred)
    label = labels[np.argmax(pred)]
    prob = np.max(pred)
    return label, prob


img = cv2.imread('./imgs/mask.jpg', cv2.IMREAD_GRAYSCALE)
get_action_class(img)

