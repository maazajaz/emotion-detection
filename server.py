import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import base64
import os

# --- Initialization ---

print("--- Starting Server ---")

# 1. Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# 2. Load your custom trained model
MODEL_PATH = "custom_emotion_model.h5"
if not os.path.exists(MODEL_PATH):
    print(f"FATAL ERROR: Model file not found at {MODEL_PATH}")
    print("Please make sure the model is trained and the file exists.")
    exit()

model = load_model(MODEL_PATH)
print(f"✅ Model '{MODEL_PATH}' loaded successfully.")

# 3. Define the emotion labels in the correct order
emotion_labels = ['Bored', 'Not Looking', 'Sleepy/Yawning', 'Interested']
print(f"✅ Emotion labels loaded: {emotion_labels}")

# 4. Load the face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("✅ Face detector loaded successfully.")


# --- API Endpoint for Prediction ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    This function is called when the frontend sends an image to the /predict URL.
    """
    # Get the image data from the request
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data received'}), 400

    # The image comes in as a base64 string, so we need to decode it
    # The header "data:image/jpeg;base64," needs to be removed first
    image_data = base64.b64decode(data['image'].split(',')[1])

    # Convert the decoded image data into a numpy array that OpenCV can use
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # --- Perform Prediction (similar to predict_custom.py) ---

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

    if len(faces) > 0:
        # We'll only process the first face found
        (x, y, w, h) = faces[0]

        # Preprocess the face for the model
        roi_gray = gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_processed = roi_resized.astype('float32') / 255.0
        roi_processed = np.expand_dims(roi_processed, axis=-1)
        roi_processed = np.expand_dims(roi_processed, axis=0)

        # Make a prediction
        predictions = model.predict(roi_processed)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotion_labels[max_index]

        # Send the prediction back to the frontend
        return jsonify({'prediction': predicted_emotion})
    else:
        # If no face is found, return a specific response
        return jsonify({'prediction': 'No Face Detected'})


# --- Start the Server ---

if __name__ == '__main__':
    # Run the Flask app on port 5000
    # host='0.0.0.0' makes it accessible from your local network
    app.run(host='0.0.0.0', port=5000, debug=True)

