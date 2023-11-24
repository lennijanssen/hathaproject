import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras_preprocessing import image
from keras.applications import resnet50
from keras.saving import load_model
from tensorflow.keras.models import load_model
import joblib
from keras.applications.inception_v3 import preprocess_input
from sklearn.preprocessing import StandardScaler
from PIL import Image
from tensorflow.keras.utils import to_categorical

# Load the pickled models and keras model
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
model = load_model('iris_nn_model.h5')

# Load the trained model
model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_shape=(3,)),
        tf.keras.layers.Softmax()])
    model.save("model.keras")
    loaded_model = tf.keras.saving.load_model("model.keras")
    x = tf.random.uniform((10, 3))
    assert np.allclose(model.predict(x), loaded_model.predict(x))
tf.keras.saving.load_model("amazing_model.keras")
model = tf.keras.models.load_model('amazing_model.keras')  # Replace with the actual path to your saved model

# Load label mapping
label_mapping = {'downdog': 0, 'goddess': 1, 'plank': 2, 'tree': 3, 'warrior2': 4}

# Load landmark data
train_df = pd.read_csv('train_landmark_all_raw.csv')
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df.iloc[:, 1:52])
label_mapping = {label: idx for idx, label in enumerate(np.unique(train_df['y_main']))}

# Placeholder for model prediction
prediction_text = ""

# Streamlit app
st.markdown("## Hatha Project 🧘‍♀️")
st.markdown("### Yoga Pose Classification")
st.write("Hatha Project is a cutting-edge application that combines the power of computer vision and neural networks to enhance your yoga practice. This innovative tool not only classifies your yoga poses in real-time using your webcam but also provides valuable insights and corrections for a more mindful and aligned practice.")


# Image Upload Section
uploaded_file = st.file_uploader("Upload your yoga pose image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Placeholder for model prediction
    prediction_text = ""

    # Display image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Display prediction
    if st.button("Classify Pose"):
        # Placeholder for model prediction
        prediction_text = "Predicting..."

        # Actual prediction logic
        # # Read the uploaded image and preprocess it
        img = Image.open(uploaded_file)
        img = img.resize((224, 224))  # Resize the image to match the model's expected sizing
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict the class
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_class_name = [class_name for class_name, idx in label_mapping.items() if idx == predicted_class[0]][0]

        # Update prediction_text
        prediction_text = f"Predicted Pose: {predicted_class_name}"

    st.write(prediction_text)
