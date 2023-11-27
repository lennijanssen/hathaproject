import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
# from project_logic.angle_comparer import angle_comparer, angle_function
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import joblib


# model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
# movenet = model.signatures['serving_default']

interpreter = tf.lite.Interpreter(model_path="models/3.tflite")
interpreter.allocate_tensors()
model = tf.keras.models.load_model('notebooks/24112023_model.h5')
scaler = joblib.load('notebooks/scaler.pkl')
label_mapping = {0: 'downdog', 1: 'goddess', 2: 'plank', 3: 'tree', 4: 'warrior2'}

# Set up
st.title("My first Streamlit app")
st.write("Hello, world")


def draw_key_points(frame, keypoints, conf_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > conf_threshold:
            cv2.circle(frame,(int(kx), int(ky)), 15, (0, 255, 0), 5)
    return frame

def get_pose(landmarks: list):
    """
    This function takes a (3,17) landmarks array and outputs the expected pose.
    """
    # Prep input before feeding to the model.
    lms_51 = np.array(landmarks).reshape(51).tolist()
    landmarks_array = np.array(landmarks).reshape(1, -1)
    landmarks_array = np.delete(landmarks_array, np.arange(2, landmarks_array.size, 3))
    landmarks_array = landmarks_array[np.newaxis, :]
    scaled_landmarks = scaler.transform(landmarks_array)

    # Feed landmarks_array to model to get softmax output
    prediction = model.predict(scaled_landmarks)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_class_name = label_mapping[predicted_class[0]]

    return str(predicted_class_name)

def callback(frame):
    image = frame.to_ndarray(format="bgr24")

    img = tf.image.resize_with_pad(np.expand_dims(image, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], input_image.numpy())

    # Run inference
    interpreter.invoke()

    # Get the output details and retrieve the keypoints with scores
    output_details = interpreter.get_output_details()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])

    # Draw the keypoints on the original frame
    image = draw_key_points(image, keypoints_with_scores, conf_threshold=0.7)
    landmarks = keypoints_with_scores[0][0]

    """ ======== Text display ========"""
    # Coordinates where the text will appear
    text_position = (50, 50)

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    line_type = 2

    text = get_pose(landmarks)

    # Put the text on the frame
    cv2.putText(image, text, text_position, font, font_scale, font_color, line_type)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(
    key="example",
    video_frame_callback=callback,
    rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
