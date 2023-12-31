import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from project_logic.angle_comparer import angle_comparer
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import joblib
from project_logic.best_poses import *
import time
from collections import deque
from PIL import Image

#Make page wide (remove default wasted whitespace)
st.set_page_config(layout="wide")

#Remove the menu button and Streamlit icon on the footer
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

#Change font to Catamaran
streamlit_style = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Catamaran:wght@100;200;300;400;500;600;700;800;900&display=swap');

            html, body, [class*="css"]  {
            font-family: 'Catamaran', sans-serif;
            }
            </style>
            """
st.markdown(streamlit_style, unsafe_allow_html=True)

# Centralize the title 'Hatha Project'
st.markdown("<h1 style='text-align: center; color: black;'>🧘‍♀️ Hatha Project 🧘‍♀️</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Supporting affordable yoga practice at home</h3>", unsafe_allow_html=True)
st.markdown("Some blurb about the project, we're awesome we're cool etc etc Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.", unsafe_allow_html=True)

# Define the layout for the 'How it works' section
col1, col2, col3 = st.columns([1,1,1])
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)
with col1:
    st.write("<p class = big-font>Step 1:</p>", unsafe_allow_html=True)
    st.markdown("User holds a yoga pose in front of the camera")
with col2:
    st.write("<p class = big-font>Step 2:</p>", unsafe_allow_html=True)
    st.markdown("Hatha Support recognizes the pose")
with col3:
    st.write("<p class = big-font>Step 3:</p>", unsafe_allow_html=True)
    st.markdown("User receives instant feedback on the pose")

interpreter = tf.lite.Interpreter(model_path="models/3.tflite")
interpreter.allocate_tensors()
model = tf.keras.models.load_model('notebooks/24112023_sub_model.h5')
scaler = joblib.load('notebooks/scaler.pkl')
label_mapping = {
    0: 'Downdog',
    1: 'Goddess',
    2: 'Plank',
    3: 'Plank',
    4: 'Tree',
    5: 'Tree',
    6: 'Warrior',
    7: 'Warrior'}
best_pose_map = {
    0: best_downdog,
    1: best_goddess,
    2: best_plank_elbow,
    3: best_plank_straight,
    4: best_tree_chest,
    5: best_tree_up,
    6: best_warrior,
    7: best_warrior}
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    # (0, 5): 'm',
    # (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_key_points(frame, keypoints, conf_threshold):
    max_dim = max(frame.shape)
    shaped = np.squeeze(np.multiply(keypoints, [max_dim,max_dim,1]))
    print(int(shaped[0][0]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > conf_threshold:
            cv2.circle(frame,(int(kx), int(ky)-80), 1, (0, 255, 0), 5)
    return frame

def draw_connections(frame, keypoints, edges, confidence_threshold):
    max_dim = max(frame.shape)
    shaped = np.squeeze(np.multiply(keypoints, [max_dim,max_dim,1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)-80), (int(x2), int(y2)-80), (255,0,0), 2)

def get_pose(landmarks: list):
    """
    This function takes a (3,17) landmarks array and returns the softmax output
    from the multiclass classification pose NN.
    """
    # Prep input before feeding to the model.
    lms_51 = np.array(landmarks).reshape(51).tolist()
    landmarks_array = np.array(lms_51).reshape(1, -1)
    landmarks_array = np.delete(landmarks_array, np.arange(2, landmarks_array.size, 3))
    landmarks_array = landmarks_array[np.newaxis, :]
    scaled_landmarks = scaler.transform(landmarks_array)

    # Feed landmarks_array to model to get softmax output
    prediction = model.predict(scaled_landmarks)
    return prediction

# This is for vertical bars and working
# def draw_bars(frame, angle_diffs, max_value=1.0, bar_width=40, bar_spacing=20):
    start_x, start_y = 50, 400  # Starting position of the first bar
    for i, angle_diff in enumerate(angle_diffs):
        # Normalize the angle difference to a value between 0 and 1
        normalized_diff = angle_diff / max_value
        bar_height = int(normalized_diff * 100)  # Scale the bar height
        end_x = start_x + bar_width
        end_y = start_y - bar_height

        # Draw the bar
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (100, 100, 255), -1)

        # Draw the text (optional)
        cv2.putText(frame, f"{angle_diff:.1f}", (start_x, start_y - bar_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Update the start_x position for the next bar
        start_x += bar_width + bar_spacing

def draw_bars(frame, angle_diffs, max_value=1.0, bar_height=28, bar_spacing=2):
    start_y, start_x = 95, 0  # Starting position of the first bar at the top-left
    for i, angle_diff in enumerate(angle_diffs):
        # Normalize the angle difference to a value between 0 and 1
        normalized_diff = angle_diff / max_value
        bar_length = int(normalized_diff * 100)  # Scale the bar length to the desired value

        # Calculate the top-left and bottom-right corners of the bar
        top_left_corner = (start_x, start_y + (bar_height + bar_spacing) * i)
        bottom_right_corner = (start_x + bar_length, start_y + bar_height + (bar_height + bar_spacing) * i)

        # Draw the bar
        cv2.rectangle(frame, top_left_corner, bottom_right_corner, (100, 100, 255), -1)

        # Draw the text (optional)
        # cv2.putText(frame, f"{angle_diff:.1f}", (start_x + bar_length + 5, start_y + bar_height * 0.7 + (bar_height + bar_spacing) * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)



# Initialize global variables for the sliding window
window_size = 30  # Number of frames to average over
angle_diff_history = deque(maxlen=window_size)
avg_percentage_diff_history = deque(maxlen=window_size)


def callback(frame):
    global angle_diff_history, avg_percentage_diff_history

    s_time = time.time()
    """ ======== 1. Movenet to get Landmarks ======== """
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
    # Draw the landmarks onto the image with threshold
    draw_key_points(image, keypoints_with_scores, conf_threshold=0.4)
    draw_connections(image, keypoints_with_scores, EDGES, 0.4)

    """ ======== 2. Pose Prediction ======== """
    pose_output = get_pose(keypoints_with_scores[0][0])
    target_pose = label_mapping[np.argmax(pose_output)]
    if np.max(pose_output) < 0.8:
        target_pose = "...still thinking..."


    """ ======== 2.1. Text and Rectangle for Pose prediction ======== """
    # Coordinates where the text will appear
    text_position = (180, 34)

    # Font settings
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1
    font_color = (0, 0, 0)  # White color
    line_type = 2

    # which text??
    text = target_pose
    text_bottom_left = text_position

    # Set the fixed-size rectangle dimensions
    box_width = 300
    box_height = 50

    # Set the top-left corner of the rectangle
    rectangle_top_left = (170, -4)  # You can change this to position the box anywhere on the image

    # Calculate the bottom-right corner of the rectangle based on the fixed size
    rectangle_bottom_right = (rectangle_top_left[0] + box_width, rectangle_top_left[1] + box_height)

    # Draw the fixed-size rectangle on the image
    rectangle_color = (0, 255, 0)  # Green color for the rectangle
    rectangle_thickness = -1  # Thickness of the rectangle borders

    cv2.rectangle(image, rectangle_top_left, rectangle_bottom_right, rectangle_color, rectangle_thickness)
    cv2.putText(image, text, text_position, font, font_scale, font_color, line_type)

    """ ======== 3. Scoring of Pose ========"""

    best = np.array(best_pose_map[np.argmax(pose_output)])
    test_angle_percentage_diff, average_percentage_diff = angle_comparer(keypoints_with_scores[0][0], best)

    angle_diff_history.append(test_angle_percentage_diff)
    avg_percentage_diff_history.append(average_percentage_diff)

    # Calculate the sliding window averages
    sliding_avg_diff = np.mean(angle_diff_history, axis=0)
    sliding_avg_percentage_diff = np.mean(avg_percentage_diff_history)


    sliding_avg_diff = np.mean(angle_diff_history, axis=0)

    # Draw the bars onto the frame
    draw_bars(image, sliding_avg_diff)


    cv2.putText(image, f"Score: {round(1-sliding_avg_percentage_diff,2)}", (210, 80), font, font_scale, font_color, line_type)

    cv2.putText(image, f"Elbow L", (0, 120), font, font_scale, font_color, line_type)
    cv2.putText(image, f"Elbow R", (0, 150), font, font_scale, font_color, line_type)
    cv2.putText(image, f"Shoulder L", (0, 180), font, font_scale, font_color, line_type)
    cv2.putText(image, f"Shoulder R", (0, 210), font, font_scale, font_color, line_type)
    cv2.putText(image, f"Hip L", (0, 240), font, font_scale, font_color, line_type)
    cv2.putText(image, f"Hip R", (0, 270), font, font_scale, font_color, line_type)
    cv2.putText(image, f"Knee L", (0, 300), font, font_scale, font_color, line_type)
    cv2.putText(image, f"Knee R", (0, 330), font, font_scale, font_color, line_type)


    # cv2.putText(image, f"Elbow L: {round(1-sliding_avg_diff[0],1)}", (0, 120), font, font_scale, font_color, line_type)
    # cv2.putText(image, f"Elbow R: {round(1-sliding_avg_diff[1],1)}", (0, 150), font, font_scale, font_color, line_type)
    # cv2.putText(image, f"Shoulder L: {round(1-sliding_avg_diff[2],1)}", (0, 180), font, font_scale, font_color, line_type)
    # cv2.putText(image, f"Shoulder R: {round(1-sliding_avg_diff[3],1)}", (0, 210), font, font_scale, font_color, line_type)
    # cv2.putText(image, f"Hip L: {round(1-sliding_avg_diff[4],1)}", (0, 240), font, font_scale, font_color, line_type)
    # cv2.putText(image, f"Hip R: {round(1-sliding_avg_diff[5],1)}", (0, 270), font, font_scale, font_color, line_type)
    # cv2.putText(image, f"Knee L: {round(1-sliding_avg_diff[6],1)}", (0, 300), font, font_scale, font_color, line_type)
    # cv2.putText(image, f"Knee R: {round(1-sliding_avg_diff[7],1)}", (0, 330), font, font_scale, font_color, line_type)

    print(f"Runtime is {round((time.time() - s_time)*1000, 2)}")
    return av.VideoFrame.from_ndarray(image, format="bgr24")

best_downdog = Image.open('mika_poses/best_downdog.jpeg')
best_goddess = Image.open('mika_poses/best_goddess.jpeg')
best_highplank = Image.open('mika_poses/best_highplank.jpeg')
best_hightree = Image.open('mika_poses/best_hightree.jpeg')
best_warrior = Image.open('mika_poses/best_warrior.jpeg')

#The 10-seconds countdown
ph = st.empty()
N = 15
for secs in range(N,0,-1):
    mm, ss = secs//60, secs%60
    ph.metric("Timer", f"{mm:02d}:{ss:02d}")
    time.sleep(1)

# Define the layout for the video feed and pose images
video_col, pose_col = st.columns([3, 1])  # Adjust the column width ratios as needed

with video_col:
    webrtc_streamer(
        key="example",
        video_frame_callback=callback,
        rtc_configuration={  # Add this line
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

# (currently not) Scrollable pose images
with pose_col:
    # Use a container and set the overflow property to allow scrolling
    with st.container():
        st.image(best_downdog, use_column_width=True)
        st.image(best_goddess, use_column_width=True)
        st.image(best_highplank, use_column_width=True)
        st.image(best_hightree, use_column_width=True)
        st.image(best_warrior, use_column_width=True)
        # Custom CSS to make the container scrollable
        st.markdown("""
            <style>
            .stContainer {
                overflow-y: auto;
                max-height: 720px;  /* Adjust the max-height to match the video feed */
            }
            </style>
            """, unsafe_allow_html=True)


# Add the footer with copyright information
st.markdown("<div style='text-align: center; color: grey;'>Copyright © The Hatha Team 2023</div>", unsafe_allow_html=True)
