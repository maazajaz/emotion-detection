import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- Load the Trained Model and Emotion Labels ---

# 1. Load the model you saved after training
model = load_model('facial_expression_model.h5')

# 2. Define the emotion labels that correspond to the model's output
# The order MUST match the order from the FER2013 dataset
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 3. Load the pre-trained Haar Cascade model for face detection from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Start Webcam Feed ---

# 4. Start capturing video from the webcam (device 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # --- Process Each Detected Face ---
    for (x, y, w, h) in faces:
        # 5. Extract the face region (Region of Interest - ROI)
        roi_gray = gray[y:y + h, x:x + w]

        # 6. Resize the face to 48x48 pixels, as required by the model
        roi_resized = cv2.resize(roi_gray, (48, 48))

        # 7. Preprocess the image for the model
        #    - Add a channel dimension (for grayscale)
        #    - Normalize pixel values to be between 0 and 1
        #    - Add a batch dimension
        roi_processed = roi_resized.astype('float32') / 255.0
        roi_processed = np.expand_dims(roi_processed, axis=-1)
        roi_processed = np.expand_dims(roi_processed, axis=0)

        # 8. Make a prediction with your trained model
        predictions = model.predict(roi_processed)

        # 9. Get the emotion with the highest probability
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotion_labels[max_index]

        # --- Display the Results ---

        # 10. Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 11. Put the predicted emotion text above the rectangle
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the final frame in a window
    cv2.imshow('Emotion Detector', frame)

    # 12. Press 'q' to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()