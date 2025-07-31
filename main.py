from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename
import uuid
import time
# Initialize Flask app
app = Flask(__name__)

# Load trained deep learning model
model = load_model('model.h5')

# Dictionary to interpret model predictions
class_dictionary = {0: 'no_truck', 1: 'truck'}

# Load video file
cap = cv2.VideoCapture('parkinglot.mp4')

# Load saved car parking positions
with open('pos.pkl', 'rb') as f:
    posList = pickle.load(f)

# Parking space dimensions (width x height)
width, height = 70, 250

# ------------------------------------
# Parking Space Detection Logic
# ------------------------------------
def checkParkingSpace(img):
    spaceCounter = 0
    imgCrops = []

    # Extract each parking region of interest (ROI)
    for pos in posList:
        x, y = pos
        imgCrop = img[y:y + height, x:x + width]
        imgResize = cv2.resize(imgCrop, (48, 48))          # Resize for model
        imgNormalized = imgResize / 255.0                  # Normalize
        imgCrops.append(imgNormalized)

    imgCrops = np.array(imgCrops)

    # Make predictions using the model
    predictions = model.predict(imgCrops)

    for i, pos in enumerate(posList):
        x, y = pos
        inID = np.argmax(predictions[i])
        label = class_dictionary[inID]
        if label == 'no_truck':
            spaceCounter += 1
    totalSpaces = len(posList)

    return img, spaceCounter, totalSpaces - spaceCounter

#video

def generate_frames():
    global cap
    while True:
        success, frame = cap.read()

        if not success:
            #Loop the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
            continue
            

        # Resize
        frame = cv2.resize(frame, (1280, 720))

        # Apply your truck detection logic here
        frame, _, _ = checkParkingSpace(frame)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(1/15)   

# ------------------------------------
# Flask Routes
# ------------------------------------

# Home page route
@app.route('/')

def index():
    return render_template('index.html')
# Status route
@app.route('/status')
def status():
    global cap
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, img = cap.read()
    if success:
        img = cv2.resize(img, (1280, 720))
        _, free_spaces, occupied_spaces = checkParkingSpace(img)
        return jsonify(free=free_spaces, occupied=occupied_spaces)
    return jsonify(free=0, occupied=0)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ------------------------------------
# Run App
# ------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
