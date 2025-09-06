import cv2
import os
import time

# --- Configuration ---
DATASET_PATH = "custom_dataset"
SAMPLES_PER_CLASS = 100  # Aim for at least 100 images per class

# --- Define the classes ---
# We map a key press to a folder name
emotions = {
    'b': '0_bored',  # Press 'b' for a bored or uninterested face
    'n': '1_not_looking',  # Press 'n' when you are not looking at the screen
    's': '2_sleepy_yawning',  # Press 's' when you are sleepy or yawning
    'i': '3_interested'  # Press 'i' for an interested, engaged face
}

# --- Setup Folders and Webcam ---
for emotion_folder in emotions.values():
    folder_path = os.path.join(DATASET_PATH, emotion_folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

print("--- Starting Data Collection ---")
print(f"Goal: {SAMPLES_PER_CLASS} images per class. Press 'q' to quit.")
print("  'b' -> Bored/Uninterested Face")
print("  'n' -> Not Looking at Screen (look left/right)")
print("  's' -> Sleepy or Yawning")
print("  'i' -> Interested and Engaged")

# --- Main Data Collection Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

    cv2.putText(frame, "Press a key: 'b', 'n', 's', 'i', or 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # We check for the key press only when a face is detected
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if chr(key) in emotions:
            emotion_key = chr(key)
            emotion_folder = emotions[emotion_key]
            folder_path = os.path.join(DATASET_PATH, emotion_folder)

            count = len(os.listdir(folder_path))
            if count < SAMPLES_PER_CLASS:
                face_roi = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face_roi, (48, 48))

                img_name = f"{emotion_folder}_{int(time.time() * 1000)}.jpg"
                save_path = os.path.join(folder_path, img_name)

                cv2.imwrite(save_path, face_resized)
                print(f"Saved {save_path} ({count + 1}/{SAMPLES_PER_CLASS})")
            else:
                print(f"'{emotion_folder}' class is full!")

    # Display the frame regardless of face detection
    cv2.imshow('Data Collection', frame)

    # Check for quit key even if no face is detected
    if len(faces) != 1:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()