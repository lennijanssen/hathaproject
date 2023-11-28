import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow_hub as hub

# Load the pre-trained model and scaler
model = tf.keras.models.load_model('24112023_model.h5')
scaler = joblib.load('scaler.pkl')
label_mapping = {0: 'downdog', 1: 'goddess', 2: 'plank', 3: 'tree', 4: 'warrior2'}

# Load MoveNet model
model_name = "movenet_lightning"
move_net_module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
input_size = 192

# Streamlit UI
st.title('Dense Neural Network (NN) Exploration')

# Header and Sub-header
st.header('Yoga Pose Recognition')
st.subheader('Upload an image to predict the yoga pose')
st.write('')

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image for MoveNet
    input_size = 192
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = tf.convert_to_tensor(image, dtype=tf.uint8)
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    input_image = input_image[..., :3]

    # input_size = 192
    # image = tf.io.read_file(uploaded_file)
    # image = tf.image.decode_jpeg(image)
    # input_image = tf.expand_dims(image, axis=0)
    # input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    # input_image = input_image[..., :3]

    # Detect landmarks using MoveNet
    move_net_model = move_net_module.signatures["serving_default"]
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = move_net_model(input_image)
    landmarks = outputs["output_0"].numpy().reshape(51).tolist()

    # Display landmarks (optional)
    st.write("Landmarks:", landmarks)

    # Reshape landmarks for further processing (if needed)
    landmarks_array = np.array(landmarks).reshape(1, -1)
    landmarks_array = np.delete(landmarks_array, np.arange(2, landmarks_array.size, 3))
    landmarks_array = landmarks_array[np.newaxis, :]
    st.write(landmarks_array.shape)

    # Button to predict the pose
    if st.button('What pose is this?'):
        # Scale landmarks using the previously trained scaler
        print(landmarks_array.shape)
        scaled_landmarks = scaler.transform(landmarks_array)

        # Make prediction using the combined features
        model = tf.keras.models.load_model('24112023_model.h5')
        prediction = model.predict(scaled_landmarks)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_class_name = label_mapping[predicted_class[0]]

        # Display the predicted yoga pose
        st.success(f"The predicted yoga pose is: {predicted_class_name}")
