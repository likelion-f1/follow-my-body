import logging
import configparser
from flask import Flask, request
from flask import render_template
from flask_socketio import SocketIO, emit
from io import StringIO, BytesIO
from base64 import b64decode, b64encode
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

import run

width = 640
height = 480

app = Flask(__name__)
# app = application
# app.run('http://127.0.0.1:8080')
socketio = SocketIO(app)
answer_mask = ''

@app.route('/')
def index():
    run.set_new_answer(height, width)
    return render_template('index.html', height=height, width=width)


@app.route('/run')
def start():
    # ini = configparser.ConfigParser()
    # ini.read('config.ini', encoding='utf-8')
    #
    # run.run(VideoProperties(ini))
    print('home')
    return 'Hello World!'


@app.route('/test', methods=['GET'])
def test():
    return "hello world!"


@app.route('/submit', methods=['POST'])
def submit():
    image = request.args.get('image')

    print(type(image))
    print(image)

    b = BytesIO(b64decode(image))
    img = Image.open(b)
    plt.imshow(img)
    return ""


def get_sample_outline():
    # 샘플 정답지
    sample_mask = cv2.imread('./imgs/mask.jpg', cv2.IMREAD_GRAYSCALE)
    sample_mask = np.array(sample_mask)

    sample_mask = cv2.resize(sample_mask,
                             dsize=(sample_mask.shape[1] // 3, sample_mask.shape[0] // 3),
                             interpolation=cv2.INTER_AREA)
    dest_mask = np.zeros((height, width), dtype=np.uint8)
    s_height, s_width = sample_mask.shape[0], sample_mask.shape[1]
    dest_mask[-20 - s_height: -20, int(width / 2 - s_width / 2):int(width / 2 - s_width / 2) + s_width] = sample_mask
    answer_outline = run.get_outline(dest_mask)
    return answer_outline


@socketio.on('connect')
def socket_connect():
    print('[socket] connection SUCCESS!')


@socketio.on('analyze')
def analyze(data_image):
    # print('[socket] analyze processing...')

    # decode and convert into image
    b = BytesIO(b64decode(data_image))
    img = Image.open(b)
    # plt.imshow(img)

    ## converting RGB to BGR, as opencv standards
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Process the image frame
    frame = run.run_with_img(frame)

    imgencode = cv2.imencode('.jpg', frame)[1]

    # base64 encode
    stringData = b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)


@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace.
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500


if __name__ == '__main__':
    # app.run()
    # global answer_mask
    # answer_mask = get_sample_outline()
    socketio.run(app)