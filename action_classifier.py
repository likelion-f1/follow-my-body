import cv2
import numpy as np
from tensorflow.keras import models

model = models.load_model('./model/action_cl_model_v1')
labels = ["batman", "ironman", "joker", 'spiderman', 'squat', 'superman', 'wonderwoman']


def get_action_class(img):
    img = cv2.resize(img, dsize=(128, 128))
    pred = model.predict(img.reshape((1, 128, 128, 1)))
    print(pred)
    label = labels[np.argmax(pred)]
    prob = np.max(pred)
    return label, prob

# img = cv2.imread('./imgs/mask.jpg', cv2.IMREAD_GRAYSCALE)
# get_action_class(img)

