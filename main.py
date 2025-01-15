import os

import cv2

from src.calibration_agents import NaiveCalibrationAgent
from src.gaze_estimation_engine import GazeEstimationEngine
from src.gaze_net import GazeNet
from tools import camera_capture, infer_point_of_regard, update_calibration_map

if __name__ == "__main__":
    net = GazeNet(filepath=os.path.join("models", "L2CSNet_gaze360.pkl"))
    ca = NaiveCalibrationAgent("yo")
    gee = GazeEstimationEngine(net, ca)
    url = "http://192.168.1.102:5000/video"
    cap = cv2.VideoCapture(url)

    filepath = os.path.join("data", "calibration_images")

    # camera_capture(cap, filepath)
    # update_calibration_map(ca, net, filepath)
    # cap = cv2.VideoCapture(url)
    # infer_point_of_regard(cap, ca, net)

    image_names = sorted(
        [
            os.path.join(filepath, f)
            for f in os.listdir(filepath)
            if f.endswith((".jpg"))
        ]
    )
    calibration_data = []
    for image in image_names:
        image_name = image.split("/")[-1]
        x, y = image_name.split("_")[1], image_name.split("_")[3].split(".")[0]
        x, y = int(x), int(y)

        frame = cv2.imread(image)

        calibration_data.append((x, y, frame))

    gee.run_calibration_steps(calibration_data)

    print("Press 'c' to capture a picture, and 'q' to quit.")
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()

        # Display the current frame
        cv2.imshow("Camera", frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            # Quit the application
            print("Quitting...")
            break
        elif key == ord("c"):
            # Capture the picture
            prediction = gee.predict_gaze_position(frame)
            print(prediction)

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
