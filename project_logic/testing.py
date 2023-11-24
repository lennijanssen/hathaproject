import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def overlay(frame):
    return frame

def live_feed():
    # Capture feed from webcam (zero)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Apply overlay
        aug_frame = overlay(frame=frame)

        # Display the feed
        cv2.imshow("Live Feed", aug_frame)

        # Break out with q key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    return None

live_feed()
