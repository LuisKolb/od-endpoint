from flask import Flask, request, Response, jsonify, render_template, flash, redirect, url_for, send_from_directory
from flask_assets import Bundle, Environment
from werkzeug.utils import secure_filename

import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import os
import time
from datetime import datetime
import json
from urllib.request import urlopen
from six import BytesIO
from base64 import encodebytes
import numpy as np
import tempfile
import requests
import uuid
import base64

#
# flask setup
#
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
assets = Environment(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['TEMPLATES_AUTO_RELOAD'] = True
css = Bundle("src/main.css", output="dist/main.css")

assets.register("css", css)
css.build()

from os import path, walk

extra_dirs = ['templates',]
extra_files = extra_dirs[:]
for extra_dir in extra_dirs:
    for dirname, dirs, files in walk(extra_dir):
        for filename in files:
            filename = path.join(dirname, filename)
            if path.isfile(filename):
                extra_files.append(filename)

print('[INFO] Started Flask App.')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#
# tensorflow setup
#
print(f'[INFO] tensorflow version: {tf.__version__}')
models_dict = {
    'none': '',
    'mobilenet_v2': 'https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1',
    'inception_resnet_v2': 'https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1',
}
model_name = 'mobilenet_v2'
model_handle = models_dict[model_name]

if model_handle:
    print(f'[INFO] loading model from tfhub: {model_handle}')
    detector = hub.load(model_handle).signatures['default']
    print(f'[INFO] model loaded.')
else:
    detector = None


#
# function definitions for image processing
#
def display_image(image):
    plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)

def download_image(url):
    pil_image = Image.open(urlopen(url))
    filename = str(uuid.uuid4()) + '.jpg'
    pil_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename), format="JPEG", quality=90)
    return filename

# expects image_data returned by BytesIO
def resize_image(path, new_width=256, new_height=256, display=False):
    #_, filename = tempfile.mkstemp(suffix=".jpg")
    pil_image = Image.open(path)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(path, format="JPEG", quality=90)
    print(f'[INFO] Resized image saved to {path}.')

    if display:
        display_image(pil_image)
    
    return path

def save_annotated_image(image, path):
    plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imsave(path, image)
    plt.close()
    print("Annotated Image saved to %s" % path)
    return path

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = BytesIO()
    pil_img.save(byte_arr, format='JPEG') # convert the PIL image to byte array
    encoded_img = base64.b64encode(byte_arr.getvalue()).hex()
    return encoded_img

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                           int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image


#
# function definitions for object detection
#
def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def run_detector(detector, image_path, output_path=''):
    img = load_img(image_path)

    converted_img = tf.image.convert_image_dtype(img, tf.float32)[
        tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key: value.numpy() for key, value in result.items()}

    #print("Found %d objects." % len(result["detection_scores"]))
    #print("Inference time: ", end_time-start_time)

    image_with_boxes = draw_boxes(
        img.numpy(), result["detection_boxes"],
        result["detection_class_entities"], result["detection_scores"])
    
    if output_path:
        save_annotated_image(image_with_boxes, output_path)

    output_dict = {
        'objects_found': len(result["detection_scores"]),
        'detection_class_entities': [e.decode('ASCII') for e in result["detection_class_entities"].tolist()],
        'detection_scores': result["detection_scores"].tolist(),
        'inference_time': end_time-start_time,
        'annotated_image_path': output_path,
        'model_used': model_handle,
    }

    return output_dict

# params:
# image_path    path of a saved (.jpg) image
# output_path   optional, string, path to save the image to
def detection_loop(filename, output = False):

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_path = resize_image(image_path, 640, 480)


    if output:
        # where results are saved
        output_path_jpg = os.path.join(app.config['UPLOAD_FOLDER'], 'output', filename + '.annotated.jpg')
        output_path_json = os.path.join(app.config['UPLOAD_FOLDER'], 'output',  filename + '.annotated.json')
    else:
        output_path_jpg = ''
        output_path_json = ''

    results_dict = run_detector(detector, image_path, output_path_jpg)

    if output_path_json:
        # save the results dict for the image
        with open(output_path_json, 'w', encoding='utf-8') as file:
            json.dump(results_dict, file, ensure_ascii=False, indent=4)

    return results_dict


#
# HTTP endpoint routing
#

# basic hello world endpoint
@app.route('/api/hello', methods=['POST', 'GET'])
def hello():
    return {'data': 'hello'}, 200

# routing http POST requests
@app.route('/api/test', methods=['POST'])
def test():
    print(request.files)
    print(request.values)
    return Response(status=200)


# routing http POST requests
@app.route('/api/detect', methods=['POST'])
def detect():
    if not detector:
        return 'no model loaded, try again later.\n', 500
    # Note that files will only contain data if the request method was POST, PUT or PATCH and the <form> that posted to the request had enctype="multipart/form-data".
    # It will be empty otherwise.

    if request.files.get('input'):
        # endpoint will be called like curl http://url:port/api/detect -X POST -F "image=@foo.jpg" [-F "output=1"]
        file = request.files['input']
        print(file.filename)
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    elif request.values.get('input'):
        # endpoint will be called like curl http://url:port/api/detect -X POST -F "input=url" [-F "output=1"]
        url = request.values.get('input')
        if not url:
            return {'data': f'no image found at the specified url:{url}\n'}, 400
        filename = str(uuid.uuid4()) + '.jpg'
        Image.open(urlopen(url)).save(os.path.join(app.config['UPLOAD_FOLDER'], filename), format="JPEG", quality=100)
        
    # (is a flag) if not empty, json and annotated images will be saved to local storage
    if request.values.get('output'):
        output = True
    else:
        output = False
    
    res_dict = detection_loop(filename, output)

    if res_dict['annotated_image_path']:
        # add the image to the response as a base64 string, in ascii encoding, to be decoded by the caller
        res_dict['annonated_image_b64'] = get_response_image(res_dict['annotated_image_path'])

    return jsonify(res_dict), 200  # return results and 200 OK


#
# ui templates rendering
#
@app.route('/', methods=['GET', 'POST'])
def landing():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            detection_loop(filename, True)

            return redirect(url_for('uploaded_file', filename= f'{filename}'))
    return render_template("index.html", model_name=model_name, model_handle=model_handle)
    
@app.route('/uploads/output/<filename>.annotated.jpg')
def uploaded_file(filename):
    full_path_jpg = os.path.join('uploads', 'output', filename + '.annotated.jpg')
    jpg_url = url_for('static', filename=full_path_jpg)

    #full_path_json = os.path.join('uploads', 'output', filename + '.annotated.json')
    #json_url = url_for('static', filename=full_path_json)

    full_path_json = os.path.join(app.config['UPLOAD_FOLDER'], 'output', f'{filename}.annotated.json')
    with open(full_path_json) as json_file:
        data = json.load(json_file)
        return render_template('display_img.html', img=jpg_url, data=data, name=filename)
    #return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'output'), filename)

if __name__ == '__main__':
    os.system('tailwindcss -i ./static/src/main.css -o ./static/dist/main.css --minify')
    app.run(debug=True, host='0.0.0.0', extra_files=extra_files)