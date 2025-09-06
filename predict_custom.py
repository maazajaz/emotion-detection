import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- Load Your Custom Trained Model and Labels ---

# 1. Load the custom model you just trained
model = load_model('custom_emotion_model.h5')

# 2. Define the new emotion labels in the correct order (0, 1, 2, 3)
# This MUST match the folders you created
emotion_labels = ['Bored', 'Not Looking', 'Sleepy/Yawning', 'Interested']

# 3. Load the face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Start Webcam Feed ---

# 4. Start capturing video from the webcam
cap = cv2.VideoCapture(0)

print("--- Starting Custom Emotion Detector ---")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        # 5. Extract and preprocess the face image exactly as you did for training
        roi_gray = gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_processed = roi_resized.astype('float32') / 255.0
        roi_processed = np.expand_dims(roi_processed, axis=-1)
        roi_processed = np.expand_dims(roi_processed, axis=0)

        # 6. Make a prediction with YOUR custom model
        predictions = model.predict(roi_processed)

        # 7. Get the predicted label
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotion_labels[max_index]

        # --- Display the Results ---

        # 8. Draw a rectangle and the predicted label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the final frame
    cv2.imshow('Custom Emotion Detector', frame)

    # 9. Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()