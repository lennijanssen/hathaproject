import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
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
import queue


# ======================== Setup and Model Loading =========================

# Make page wide (remove default wasted whitespace)
print('setting the page config')
st.set_page_config(layout="wide")

#Remove the menu button and Streamlit icon on the footer
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
print('hiding default menu')
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


# Load Model and Scaler
interpreter = tf.lite.Interpreter(model_path="models/3.tflite")
interpreter.allocate_tensors()
model = tf.keras.models.load_model('notebooks/24112023_sub_model.h5')
scaler = joblib.load('notebooks/scaler.pkl')

# Define necessary dictionaries
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
landmark_dict = {
    'landmarks_left_elbow': (9, 7, 5),
    'landmarks_right_elbow': (10, 8, 6),
    'landmarks_left_shoulder': (11, 5, 7),
    'landmarks_right_shoulder': (12, 6, 8),
    'landmarks_hip_left': (13, 11, 5),
    'landmarks_hip_right': (14, 12, 6),
    'landmarks_left_knee': (15, 13, 11),
    'landmarks_right_knee': (16, 14, 12)}
lm_list = list(landmark_dict.keys())
lm_points = list(landmark_dict.values())
joint_dict = {'landmarks_left_elbow': 'left elbow',
              'landmarks_right_elbow': 'left elbow',
              'landmarks_left_shoulder': 'left shoulder',
              'landmarks_right_shoulder': 'right shoulder',
              'landmarks_hip_left': 'left hip',
              'landmarks_hip_right': 'hip right',
              'landmarks_left_knee': 'left knee',
              'landmarks_right_knee': 'right knee'
              }


# Xx
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

# ==================== Functions definition and Variables =====================

# Defining functions for the
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

def draw_bars(frame, angle_diffs, max_value=1.0, bar_height=28, bar_spacing=2, max_bar_length=200):
    start_y, start_x = 95, 0  # Starting position of the first bar at the top-left
    for i, angle_diff in enumerate(angle_diffs):
        # Normalize the angle difference to a value between 0 and 1
        normalized_diff = angle_diff / max_value
        bar_length = int(normalized_diff * max_bar_length)  # Scale the bar length to the desired value

        # Ensure the bar length does not exceed max_bar_length
        bar_length = min(bar_length, max_bar_length)

        # Calculate the top-left and bottom-right corners of the bar
        top_left_corner = (start_x, start_y + (bar_height + bar_spacing) * i)
        bottom_right_corner = (start_x + bar_length, start_y + bar_height + (bar_height + bar_spacing) * i)

        # Draw the bar
        cv2.rectangle(frame, top_left_corner, bottom_right_corner, (100, 100, 255), -1)

        # Draw the text (optional)
        # cv2.putText(frame, f"{angle_diff:.1f}", (start_x + bar_length + 5, start_y + bar_height * 0.7 + (bar_height + bar_spacing) * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# Initialize global variables for the sliding window
window_size = 2  # Number of frames to average over
score_angles_history = deque(maxlen=window_size)
average_score_history = deque(maxlen=window_size)
angle_diff_history = deque(maxlen=window_size)
avg_percentage_diff_history = deque(maxlen=window_size)


# Defining the callback to create video and overlay
def callback(frame):
    global angle_diff_history, avg_percentage_diff_history, score_angles_history, average_score_history

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
    # print(keypoints_with_scores[0][0][:, :2])
    # print(type(keypoints_with_scores))


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
    test_angle_percentage_diff, average_percentage_diff, score_angles, score_angles_unscaled, average_score = angle_comparer(keypoints_with_scores[0][0][:, :2], best)

    best = np.array(best_pose_map[np.argmax(pose_output)])
    test_angle_percentage_diff, average_percentage_diff, score_angles, score_angles_unscaled, average_score = angle_comparer(keypoints_with_scores[0][0][:, :2], best)
    index_of_worst = test_angle_percentage_diff.index(max(test_angle_percentage_diff))
    worst_points = lm_points[index_of_worst]
    result_queue.put(lm_list[index_of_worst])

    # cv2.putText(image, f"Score (avg): {test_angle_percentage_diff}", (50, 100), font, font_scale, font_color, line_type)

    print(f"Runtime is {round((time.time() - s_time)*1000, 2)}")

    worst_kps = []
    for i in lm_points[index_of_worst]:
        worst_kps.append((np.squeeze(keypoints_with_scores)[i]).tolist())

    worst_edges = {
    (worst_points[0], worst_points[1]): None,
    (worst_points[1], worst_points[2]): None,
    }

    result_queue.put(worst_edges)


    # Draw the landmarks onto the image with threshold
    draw_key_points(image, worst_kps, conf_threshold=0.2)
    draw_connections(image, keypoints_with_scores, worst_edges, 0.5)
    mirrored_image = cv2.flip(image, 1)
    cv2.rectangle(mirrored_image, rectangle_top_left, rectangle_bottom_right, rectangle_color, rectangle_thickness)
    cv2.putText(mirrored_image, text, text_position, font, font_scale, font_color, line_type)

    return av.VideoFrame.from_ndarray(mirrored_image, format="bgr24")


    print(f"Runtime is {round((time.time() - s_time)*1000, 2)}")
    return av.VideoFrame.from_ndarray(image, format="bgr24")


# ==================== Actual UI output =====================

best_downdog = Image.open('mika_poses/best_downdog.jpeg')
best_highplank = Image.open('mika_poses/best_highplank.jpeg')
best_hightree = Image.open('mika_poses/best_hightree.jpeg')
best_goddess = Image.open('mika_poses/best_goddess.jpeg')
best_warrior = Image.open('mika_poses/best_warrior.jpeg')

# Show the poses with the loading spinner
pose_col_1, pose_col_2, pose_col_3, pose_col_4, pose_col_5 = st.columns([1, 1, 1, 1, 1])

with pose_col_1:
    with st.container():
        st.image(best_downdog, use_column_width=True, caption='Downward Facing Dog')

with pose_col_2:
    with st.container():
        st.image(best_highplank, use_column_width=True, caption='High Plank')

with pose_col_3:
    with st.container():
        st.image(best_hightree, use_column_width=True, caption='High Tree')

with pose_col_4:
    with st.container():
        st.image(best_goddess, use_column_width=True, caption='Goddess')

with pose_col_5:
    with st.container():
        st.image(best_warrior, use_column_width=True, caption='Warrior')

webrtc_streamer(
    key="example",
    video_frame_callback=callback,
    rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False}  # Disable audio
)
# main()

labels_placeholder = st.empty()
angle_perc = st.empty()
timecount =  st.empty()

# Sample variables to be shown, to be replaced with real-time data
sample_pose = "Warrior"
sample_score = "86%"
sample_body_part = "Left elbow"

# Add the loading spinner around the section where results are updated
with st.spinner('Analyzing pose...'):
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("""### Pose Analysis""")
    col2.metric("Pose", sample_pose)
    col3.metric("Score", sample_score)
    col4.metric("You need to fix", sample_body_part)

# Learn More Section
with st.expander("""More about the yoga poses and the benefits 🧘"""):
    st.write("""
        - **Downward-Facing Dog**:
        A quintessential yoga pose, Downward-Facing Dog provides a rejuvenating stretch for the entire body. It's known for its ability to calm the mind, lengthen the spine, strengthen the upper body, and stimulate blood flow to the brain. This pose is often used as a resting posture in between more challenging poses.

        - **Plank**:
        Foundational pose that strengthens the arms, wrists, and spine while toning the abdomen. It's a full-body workout that requires energy, engagement, and stability, and it's an essential component for building core strength and resilience.

        - **Tree**:
        Strengthens the legs, opens the hips, and cultivates concentration and clarity of mind. By mimicking the steady stance of a tree, practitioners learn to root themselves firmly to the ground, promoting a sense of grounding and balance.

        - **Goddess**:
        Dynamic standing posture that ignites the fires of the inner thighs, hips, and chest. As a pose that encourages powerful energy flow, it's excellent for improving circulation and energizing the body. It also fosters a sense of inner strength and empowerment.

        - **Warrior**:
        Helps build focus, power, and stability. This powerful stretch for your thighs and shoulders increases stamina as well.
    """)

# Add the footer with copyright information
st.markdown("<div style='text-align: center; color: grey;'>Copyright © The Hatha Team 2023</div>", unsafe_allow_html=True)

while True:
    s_time = time.time()
    worst = result_queue.get()
    result = max(result_queue.get())
    labels_placeholder.write(f"results: {result}")
    angle_perc.write(f"FIX YOUR {worst}")
    timecount.write(f"Runtime is {round((time.time() - s_time)*1000, 2)}")
