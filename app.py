import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
# from project_logic.angle_comparer import angle_comparer, angle_function
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub


model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
movenet = model.signatures['serving_default']

interpreter = tf.lite.Interpreter(model_path="models/3.tflite")
interpreter.allocate_tensors()

st.title("My first Streamlit app")
st.write("Hello, world")

threshold1 = st.slider("Threshold1", min_value=0, max_value=1000, step=1, value=100)
threshold2 = st.slider("Threshold2", min_value=0, max_value=1000, step=1, value=200)

def draw_key_points(frame, keypoints, conf_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > conf_threshold:
            cv2.circle(frame,(int(kx), int(ky)), 15, (0, 255, 0), 5)
    return frame


def callback(frame):
    image = frame.to_ndarray(format="bgr24")

    image = image[0:300,:,:]

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

    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(
    key="example",
    video_frame_callback=callback,
    rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
