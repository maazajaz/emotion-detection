import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Change this line
DATASET_PATH = "." # The "." means "the current folder"
MODEL_SAVE_PATH = "custom_emotion_model.h5"
# The exact folder names the script expects to find
REQUIRED_FOLDERS = ['0_bored', '1_not_looking', '2_sleepy_yawning', '3_interested']
MIN_IMAGES_PER_CLASS = 10  # Minimum images required in each folder to start training


# --- Step 1: Verify the Dataset Before Doing Anything Else ---

def verify_dataset(dataset_path):
    """
    Checks if the dataset is complete and ready for training.
    """
    print("--- Verifying Dataset ---")
    is_valid = True
    for folder_name in REQUIRED_FOLDERS:
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print(f"❌ ERROR: Folder '{folder_name}' is missing.")
            is_valid = False
            continue

        image_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        if image_count < MIN_IMAGES_PER_CLASS:
            print(
                f"❌ ERROR: Folder '{folder_name}' has only {image_count} images. Please collect at least {MIN_IMAGES_PER_CLASS}.")
            is_valid = False
        else:
            print(f"✅ Found {image_count} images in '{folder_name}'.")

    if not is_valid:
        print("\nDataset is not ready. Please run collect_data.py and add images to the folders marked with ERROR.")
    else:
        print("\nDataset verification successful. Ready to train.")

    return is_valid


# --- Step 2: Load Images (only if verification passes) ---

def load_custom_data(dataset_path):
    print(f"\nLoading images from {dataset_path}...")
    images = []
    labels = []

    for class_folder in REQUIRED_FOLDERS:
        class_path = os.path.join(dataset_path, class_folder)
        label = int(class_folder.split('_')[0])

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)
                labels.append(label)

    print(f"Loaded {len(images)} total images.")
    return np.array(images), np.array(labels)


# --- Step 3: Preprocess and Train ---
# (The rest of the code is the same as before)

def preprocess_for_training(X, y):
    print("Preprocessing data for training...")
    X_normalized = X.astype('float32') / 255.0
    X_reshaped = np.expand_dims(X_normalized, axis=-1)

    num_classes = len(np.unique(y))
    y_categorical = to_categorical(y, num_classes=num_classes)

    print("Images shape:", X_reshaped.shape)
    print("Labels shape:", y_categorical.shape)

    return X_reshaped, y_categorical, num_classes


def train_model(X, y, num_classes):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    print("\n--- Starting Model Training ---")
    model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_val, y_val),
        verbose=1
    )
    print("--- Model Training Finished ---")

    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved as '{MODEL_SAVE_PATH}'")


# --- Main Execution ---
if __name__ == "__main__":
    if verify_dataset(DATASET_PATH):
        X_data, y_data = load_custom_data(DATASET_PATH)
        X_processed, y_processed, num_classes = preprocess_for_training(X_data, y_data)
        train_model(X_processed, y_processed, num_classes)

