import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os


# --- Step 1: Load and Preprocess the Dataset ---

def load_and_preprocess_data(csv_path):
    """
    Loads the FER2013 dataset from a CSV file and preprocesses it for training.
    """
    print("Loading dataset...")
    # Load the dataset using pandas
    data = pd.read_csv(csv_path)
    print("Dataset loaded successfully.")
    print(data.head())
    print("\nEmotion labels:", data["emotion"].unique())
    print("\nDataset usage counts:\n", data["Usage"].value_counts())

    # --- Convert Pixel Strings to Images ---
    print("\nPreprocessing pixel data...")

    # This function converts a string of pixels into a 48x48 normalized image array
    def preprocess_pixels(pixels_str):
        pixels = np.array(pixels_str.split(), dtype="float32")
        # Normalize pixel values to be between 0 and 1
        return pixels.reshape(48, 48, 1) / 255.0

    # Apply the function to the 'pixels' column
    X = np.array([preprocess_pixels(p) for p in data["pixels"]])

    # --- One-Hot Encode Labels ---
    # Convert the numerical emotion labels (0-6) into a one-hot encoded format
    num_classes = 7
    y = to_categorical(data["emotion"], num_classes=num_classes)

    print("\nData shapes:")
    print("Images (X) shape:", X.shape)
    print("Labels (y) shape:", y.shape)

    return X, y


# --- Step 2: Split Data into Training and Testing Sets ---

def split_data(X, y):
    """
    Splits the data into training and testing sets.
    """
    print("\nSplitting data into training and testing sets...")

    # We use stratify=y to ensure both train and test sets have a similar distribution of emotions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    return X_train, X_test, y_train, y_test


# --- Step 3: Build the CNN Model ---

def build_model(input_shape, num_classes):
    """
    Builds, compiles, and returns the CNN model.
    """
    print("\nBuilding the CNN model...")

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        # The final layer must have the same number of neurons as there are classes (7 emotions)
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Model built and compiled successfully.")
    model.summary()
    return model


# --- Main Execution ---

if __name__ == "__main__":
    # Define the path to your dataset
    # Make sure the 'data' folder is in your PyCharm project root
    DATASET_PATH = os.path.join("data", "fer2013.csv")

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("Please make sure 'fer2013.csv' is inside a 'data' folder in your project directory.")
    else:
        # 1. Load and process data
        X, y = load_and_preprocess_data(DATASET_PATH)

        # 2. Split data
        X_train, X_test, y_train, y_test = split_data(X, y)

        # 3. Build model
        # The input shape is (48, 48, 1) for 48x48 grayscale images
        # The number of classes is 7 for the 7 emotions in the dataset
        model = build_model(input_shape=(48, 48, 1), num_classes=7)

        # 4. Train the model
        print("\n--- Starting Model Training ---")
        # A batch size of 64 is common for this dataset
        # We'll train for 30 epochs, but you can increase this for better accuracy
        history = model.fit(
            X_train, y_train,
            batch_size=64,
            epochs=30,
            validation_data=(X_test, y_test),
            verbose=1
        )
        print("--- Model Training Finished ---")

        # 5. Evaluate the model
        print("\n--- Evaluating Model Performance ---")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Test Loss: {loss:.4f}")

        # 6. Save the trained model for later use
        model.save("facial_expression_model.h5")
        print("\nModel saved as 'facial_expression_model.h5'")