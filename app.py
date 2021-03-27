from flask import Flask, request, redirect, render_template, jsonify
import io
import numpy as np
import cv2
from detector import Detector
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def upload_photo():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def inference():
    if request.method == 'POST' and 'photo' in request.files:
        photo = request.files['photo']
        in_memory_file = io.BytesIO()
        photo.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img = cv2.imdecode(data, color_image_flag)
        d = Detector()
        output = d.process_image(img)

        print(output)
        return output

# start flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)