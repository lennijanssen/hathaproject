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
from PIL import Image
import queue

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
st.markdown("""
<style>
.med-font {
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)
with col1:
    st.write("<p class = big-font>Step 1:</p>", unsafe_allow_html=True)
    st.markdown("<p class = med-font>User holds a yoga pose in front of the camera</p>", unsafe_allow_html=True)
with col2:
    st.write("<p class = big-font>Step 2:</p>", unsafe_allow_html=True)
    st.markdown("<p class = med-font>Hatha Support recognizes the pose</p>", unsafe_allow_html=True)
with col3:
    st.write("<p class = big-font>Step 3:</p>", unsafe_allow_html=True)
    st.markdown("<p class = med-font>User receives instant feedback on the pose</p>", unsafe_allow_html=True)

# Set up
interpreter = tf.lite.Interpreter(model_path="models/3.tflite")
interpreter.allocate_tensors()
model = tf.keras.models.load_model('notebooks/24112023_sub_model.h5')
scaler = joblib.load('notebooks/scaler.pkl')
label_mapping = {
    0: 'downdog',
    1: 'goddess',
    2: 'plank_elbow',
    3: 'plank_straight',
    4: 'tree_chest',
    5: 'tree_up',
    6: 'warrior2',
    7: 'warrior2'}
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
    (0, 5): 'm',
    (0, 6): 'c',
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

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def draw_key_points(frame, keypoints, conf_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    print(int(shaped[0][0]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > conf_threshold:
            cv2.circle(frame,(int(kx), int(ky)), 1, (0, 255, 0), 5)
    return frame

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

def draw_connections(frame, keypoints, edges, confidence_threshold):
    max_dim = max(frame.shape)
    shaped = np.squeeze(np.multiply(keypoints, [max_dim,max_dim,1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)-80), (int(x2), int(y2)-80), (255,0,0), 2)


def callback(frame):
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
    draw_key_points(image, keypoints_with_scores, conf_threshold=0.5)
    draw_connections(image, keypoints_with_scores, EDGES, 0.5)

    """ ======== 2. Pose Prediction ======== """
    pose_output = get_pose(keypoints_with_scores[0][0])
    target_pose = label_mapping[np.argmax(pose_output)]
    if np.max(pose_output) < 0.8:
        target_pose = "...still thinking..."

    # Settings for text to show predicted pose
    text_position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2
    cv2.putText(image, str(target_pose), text_position, font, font_scale, font_color, line_type)
    result_queue.put(keypoints_with_scores[0][0])

    """ ======== 3. Scoring of Pose ========"""

    best = np.array(best_pose_map[np.argmax(pose_output)])
    test_angle_percentage_diff, average_percentage_diff = angle_comparer(keypoints_with_scores[0][0], best)

    cv2.putText(image, f"Score (avg): {test_angle_percentage_diff}", (50, 100), font, font_scale, font_color, line_type)

    print(f"Runtime is {round((time.time() - s_time)*1000, 2)}")
    return av.VideoFrame.from_ndarray(image, format="bgr24")

# #The 15-seconds countdown functions
# def main():
#     if st.button("Start Timer"):
#         start_timer()

# def start_timer():
#     ph = st.empty()
#     N = 15
#     while True:
#         for secs in range(N, 0, -1):
#             mm, ss = secs // 60, secs % 60
#             ph.metric("Timer", f"{mm:02d}:{ss:02d}")
#             time.sleep(1)
#         ph.empty()  # Clear the timer display

best_downdog = Image.open('mika_poses/best_downdog.jpeg')
best_goddess = Image.open('mika_poses/best_goddess.jpeg')
best_highplank = Image.open('mika_poses/best_highplank.jpeg')
best_hightree = Image.open('mika_poses/best_hightree.jpeg')
best_warrior = Image.open('mika_poses/best_warrior.jpeg')

# Initialize session state variables if they don't exist
if 'result' not in st.session_state:
    st.session_state['result'] = None

# Define a function to update the session_state with the result and rerun the app
def update_state(result):
    st.session_state['result'] = result
    st.experimental_rerun()  # Use Streamlit's experimental rerun feature
    
st.markdown("### Pose Analysis (Placeholder)")
col1, col2 = st.columns(2)
col1.metric("Pose Name", "Warrior")
col2.metric("Score", "80%")

# Define the layout for the video feed and pose images
video_col, pose_col = st.columns([3, 1])  # Adjust the column width ratios as needed

# # USING SESSION STATE + IF
# with video_col:
#     # The live video feed remains unchanged
#     webrtc_ctx = webrtc_streamer(
#         key="example",
#         video_frame_callback=callback,
#         rtc_configuration={
#             "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#         }
#     )

#     # Check if the WebRTC connection is up and there's a new result to display
#     if webrtc_ctx.video_processor:
#         if not result_queue.empty():
#             result = result_queue.get()
#             update_state(result)
#             # Update the session state with the new result
#             st.session_state['result'] = result

#     # Display the result below the video feed
#     st.markdown("### Pose Analysis")
#     # Use the session_state to display the result
#     if 'result' in st.session_state and st.session_state['result'] is not None:
#         st.write(st.session_state['result'])
#     else:
#         st.write("No pose analysis results to display yet.")

# USING WHILE LOOP
with video_col:
    # The live video feed remains unchanged
    webrtc_streamer(
        key="example",
        video_frame_callback=callback,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
    # labels_placeholder = st.empty()
    # timecount =  st.empty()
    # while True:
    #     s_time = time.time()
    #     result = result_queue.get()
    #     labels_placeholder.write(result)
    #     timecount.write(f"Runtime is {round((time.time() - s_time)*1000, 2)}")

# Scrollable pose images
with pose_col:
    # Use a container and set the overflow property to allow scrolling
    with st.container():
        st.image(best_downdog, use_column_width=True)
        st.image(best_goddess, use_column_width=True)
        st.image(best_highplank, use_column_width=True)
        st.image(best_hightree, use_column_width=True)
        st.image(best_warrior, use_column_width=True)

# #To make the while loop work without blocking the rest of the page to execute, by placing it at the end of the page but still not working
# st.markdown("### Pose Analysis")
# labels_placeholder = st.empty()
# timecount =  st.empty()
# while True:
#     s_time = time.time()
#     result = result_queue.get()
#     labels_placeholder.write(result)
#     timecount.write(f"Runtime is {round((time.time() - s_time)*1000, 2)}")

# Add the footer with copyright information
st.markdown("<div style='text-align: center; color: grey;'>Copyright © The Hatha Team 2023</div>", unsafe_allow_html=True)
