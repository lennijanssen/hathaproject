import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from project_logic.angle_comparer import angle_comparer
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from project_logic.best_poses import *
from PIL import Image
import base64


# Container for the entire application
app_container = st.container()

with app_container:
    # ======================== Setup and Model Loading =========================

    # # Make page wide (remove default wasted whitespace)
    # print('setting the page config')
    # st.set_page_config(layout="wide")

    # Remove the menu button and Streamlit icon on the footer
    hide_default_format = """
           <style>
           #MainMenu {visibility: hidden; }
           footer {visibility: hidden;}
           </style>
           """
    print('hiding default menu')
    st.markdown(hide_default_format, unsafe_allow_html=True)

    # Change font to Catamaran
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

    # Replace the lorem_ipsum text with a smaller main image
    team_image = Image.open('mika_poses/hatha-team.jpeg')
    st.image(team_image, use_column_width=True)

    # Define the layout for the 'How it works' section
    col1, col2, col3 = st.columns([1, 1, 1])
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

    # Use Markdown to create a link button
    st.markdown(
        """
        <div style="text-align: center;">
            <a href="/app_main.py" target="_blank" style="text-decoration: none;">
                <button style="font-size: 20px; padding: 10px 20px; border-radius: 10px; background-color: black; color: white;">Get Started 🧘‍♂️</button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    # Add the footer with copyright information
    st.markdown("<div style='text-align: center; color: grey;'>Copyright © The Hatha Team 2023</div>", unsafe_allow_html=True)
