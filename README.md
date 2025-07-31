# Truck Parking Spot Counting System 🚚🅿️

This project detects and counts available truck parking spots in a parking lot using computer vision and deep learning. It processes video footage of a parking lot, analyzes each parking space using a trained CNN model, and displays the number of free and occupied spots in real time on a web interface.

---

## 🚀 Features

- Real-time video processing of parking lot footage
- Region of Interest (ROI) cropping of individual parking spots
- Deep learning model (`model.h5`) trained to classify each spot as **truck** or **no truck**
- Displays live counts of **free** and **occupied** parking spots
- Flask-based web app with live video feed and JSON status endpoint

---

## 🛠️ Setup & Installation

1. **Clone the repository:**
```bash
git clone https://github.com/M-dorra/truck_parking.git
cd truck-parking-spot-count
```
2. **Install dependencies:**

```bash
pip install flask opencv-python tensorflow numpy
```
3. **Train the model**(⚠️ Required — model file not included):

The pre-trained model.h5 file is not included due to file size limitations.
👉 To generate it, run the notebook below:

```bash
jupyter notebook training.ipynb
```

4. **Run the Flask app:**
```bash
python main.py
  ```
5. **Open your browser at:** http://127.0.0.1:5000/
   
---

## ⚙️ How It Works
- The Flask app streams the video from parkinglot.mp4.
- For each frame, it crops the predefined parking spot areas (from pos.pkl).
- Each cropped spot is resized and normalized before being fed into the CNN model.
- The model predicts whether the spot contains a truck or is free.
- The app counts and displays free and occupied spots.
- The video feed with overlay can be viewed live in the browser.
- A JSON endpoint /status provides the count of free and occupied spots.
  
---

## 📊 Model Training & Data Collection
- data_coll.py helps to select and save ROI coordinates of parking spots.
- Training dataset folders:
  - dataset/trucks/ — images of occupied spots
  - dataset/no_trucks/ — images of empty spots
- Train your model using these datasets and save as model.h5.

---

## 🧩 Notes & Tips
- Adjust parking spot ROI size (width, height) in main.py if needed.
- Make sure video resolution and ROI coordinates match.
- To retrain the model, collect balanced datasets of truck/no-truck images.
- You can extend the app to process webcam input or IP cameras.

---

## 📷 Demo

<img width="1021" height="815" alt="Screenshot 2025-07-27 110016" src="https://github.com/user-attachments/assets/0cfd8494-26de-4036-935a-d238ac83e823" />

<img width="1034" height="817" alt="Screenshot 2025-07-27 110039" src="https://github.com/user-attachments/assets/e5fe03c5-4285-4d18-913b-1d73774bc921" />
