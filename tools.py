import os
import time
from datetime import datetime
import cv2
from l2cs import Pipeline, render
import torch

from src.gaze_net import GazeNet
from src.calibration_agents import NaiveCalibrationAgent

def camera_capture(cap, filepath):

    print("Press 'c' to capture a picture, and 'q' to quit.")
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()

        # Display the current frame
        cv2.imshow('Camera', frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Quit the application
            print("Quitting...")
            break
        elif key == ord('c'):
            # Capture the picture
            print("Picture captured! Please enter coordinates.")
            x = input("Enter x coordinate: ")
            y = input("Enter y coordinate: ")
            # Save the picture
            filename = f"x_{x}_y_{y}.jpg"
            cv2.imwrite(os.path.join(filepath, filename), frame)
            print(f"Picture saved as {filename}")

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def update_calibration_map(ca : NaiveCalibrationAgent, net : GazeNet, filepath):
    image_names = sorted([os.path.join(filepath, f) for f in os.listdir(filepath) if f.endswith(('.jpg'))])

    for image in image_names:
        
        image_name = image.split("/")[-1]
        x, y = image_name.split("_")[1], image_name.split("_")[3].split(".")[0]
        x, y = int(x), int(y)
        
        frame = cv2.imread(image)
        _, _, theta, phi = net.predict_gaze_vector(frame)

        ca.calibration_step(x, y, theta, phi)
    
def infer_point_of_regard(cap, ca : NaiveCalibrationAgent, net : GazeNet):
    print("Press 'c' to capture a picture, and 'q' to quit.")
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()

        # Display the current frame
        cv2.imshow('Camera', frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Quit the application
            print("Quitting...")
            break
        elif key == ord('c'):
            # Capture the picture
            _, _, theta, phi = net.predict_gaze_vector(frame)
            prediction = ca.calculate_point_of_regard(theta, phi)
            print(prediction) 

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

        
def render_gaze_vector(frame):


    gaze_pipeline = Pipeline(
        weights='L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device=torch.device('cpu')  # or 'gpu'
    )

    print("Got image!")
    # Process frame and visualize
    results = gaze_pipeline.step(frame)

    
    frame = render(frame, results)

    cv2.imshow("Captured Frame", frame)  # Display the image

    # Non-blocking loop
    while True:
        if cv2.getWindowProperty("Captured Frame", cv2.WND_PROP_VISIBLE) < 1:
            break  # Exit loop when window is closed
        cv2.waitKey(100)  # Check every 100 ms

    # camera.release()
    cv2.destroyAllWindows()
    print("Script ended, but image window can stay open.")