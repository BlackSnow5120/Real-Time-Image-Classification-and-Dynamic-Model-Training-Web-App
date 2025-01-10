import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Initialize global variables
data_folder = "data"
model_path = "trained_model.h5"
image_size = (256, 256)

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

model = None  # Placeholder for the model

# Helper functions

def train_model():
    image_list = []
    label_list = []

    folder_path = "data"

    # Dynamically generate the label dictionary based on folder names
    class_names = os.listdir(folder_path)
    class_names = [cls for cls in class_names if os.path.isdir(os.path.join(folder_path, cls))]  # Only folders

    label_dict = {class_name: idx for idx, class_name in enumerate(class_names)}

    # Loop through the class folders to load images
    for class_name in class_names:
        class_folder = os.path.join(folder_path, class_name)
        image_files = os.listdir(class_folder)
        for image_file in image_files:
            img = Image.open(os.path.join(class_folder, image_file))
            img = img.convert("RGB").resize(image_size)  # Resize to 256x256
            img_array = np.array(img) / 255.0  # Normalize the image
            image_list.append(img_array)
            label_list.append(class_name)

    # Convert labels to numeric labels using label_dict
    numeric_labels = [label_dict[label] for label in label_list]
    num_classes = len(label_dict)  # Number of unique classes
    label = to_categorical(numeric_labels, num_classes=num_classes)

    # Convert lists to numpy arrays
    image_list = np.array(image_list)

    # Model definition
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')  # Softmax for multiclass classification
    ])

    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    # Training the model
    model.fit(image_list, label, batch_size=32, epochs=8)
    st.write("Training completed!")

    # Save the trained model
    model.save(model_path)
    st.write("Model saved to disk.")
    return model

def load_or_train_model():
    """Load the model if exists, otherwise train and save it."""
    global model
    if os.path.exists(model_path):
        st.write("Loading saved model...")
        model = tf.keras.models.load_model(model_path)
    else:
        st.write("Training model for the first time...")
        model = train_model()
        st.write("Model trained and saved!")

def preprocess_image(img):
    """Preprocess an image for prediction."""
    if isinstance(img, np.ndarray):  # If the image is a numpy array (OpenCV frame)
        img = Image.fromarray(img)  # Convert NumPy array to PIL Image
    img = img.convert("RGB").resize(image_size)  # Resize to 256x256
    img_array = np.array(img) / 255.0 
    return img_array

def predict_image(image):
    """Predict the class of an image."""
    image_array = preprocess_image(image)
    predictions = model.predict(tf.expand_dims(image_array, axis=0))
    class_idx = np.argmax(predictions)  # Get the class index
    class_names = os.listdir("data")
    class_names = [cls for cls in class_names if os.path.isdir(os.path.join("data", cls))]  # Filter class folders
    class_name = class_names[class_idx]  # Map index to class name
    return class_name, predictions[0]

def capture_continuous_images(class_name, capture_duration=10, delay=0.5):
    """Capture images continuously for the specified duration (in seconds) with a delay between frames."""
    class_folder = os.path.join(data_folder, class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
        return
    
    # Start the timer
    start_time = time.time()
    image_count = 0
    
    while time.time() - start_time < capture_duration:
        ret, frame = cap.read()
        if ret:
            image_count += 1
            image_path = os.path.join(class_folder, f"{class_name}_{image_count}.jpg")
            cv2.imwrite(image_path, frame)  # Save the captured frame as an image
            st.write(f"Captured image {image_count} at {time.time() - start_time:.2f} seconds.")
            
            # Add a delay to slow down the frame capture rate
            time.sleep(delay)  # Delay between captures (in seconds)
        else:
            st.error("Failed to capture image.")
            break
    
    cap.release()
    st.write(f"Captured {image_count} images for the class '{class_name}'.")

# Streamlit UI
st.title("Image Classification App")
st.sidebar.title("Menu")

# Load or train the model
load_or_train_model()

# Menu Options
menu = st.sidebar.radio("Choose an action", ["Home", "Capture Image", "Upload Image", "Register New Class", "Retrain Model"])

if menu == "Home":
    st.write("Welcome to the Image Classification App!")
    st.write("You can upload or capture images for prediction, or register new classes dynamically.")

elif menu == "Capture Image":
    st.header("Capture an Image")

    # Add a button to capture the image
    if st.button("Capture Image"):
        # Open the camera once
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                st.image(frame, caption="Captured Image", use_column_width=True)
                # Store the captured frame in a session state to use later for prediction
                st.session_state.frame = frame
        else:
            st.error("Unable to access the camera.")
    
    # If there's a captured frame in the session state, show the predict button
    if "frame" in st.session_state:
        if st.button("Predict"):
            class_name, confidence = predict_image(st.session_state.frame)  # Use the captured image for prediction
            st.write(f"Predicted Class: {class_name}, Confidence: {confidence}")


elif menu == "Upload Image":
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict"):
            class_name, confidence = predict_image(image)
            st.write(f"Predicted Class: {class_name}, Confidence: {confidence}")

elif menu == "Register New Class":
    st.header("Register a New Class")
    class_name = st.text_input("Enter the class name")
    if st.button("Start Capturing for 10 Seconds"):
        if class_name:
            st.write(f"Starting image capture for class: {class_name}...")
            capture_continuous_images(class_name, capture_duration=10, delay=0.1)  # Capture images for 10 seconds with delay
            st.write("Capturing completed. Retraining the model with the new images...")
            model = train_model()  # Retrain the model with the new class
            model.save(model_path)  # Save the updated model
            st.write("Retraining completed and model saved!")
        else:
            st.error("Please enter a class name first.")

elif menu == "Retrain Model":
    st.header("Retrain the Model")
    st.write("Retraining the model with new data...")
    model = train_model()  # Retrain the model with the current dataset
    model.save(model_path)  # Save the updated model
    st.write("Retraining completed and model saved!")
