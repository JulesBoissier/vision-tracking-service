import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile

from src.calibration_agents import NaiveCalibrationAgent
from src.gaze_estimation_engine import GazeEstimationEngine
from src.gaze_net import GazeNet

# Global variable for the GazeEstimationEngine instance
gaze_engine = None

# Dictionary to hold the GazeEstimationEngine instance
resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the GazeEstimationEngine
    nca = NaiveCalibrationAgent(db_path=None)
    gn = GazeNet(filepath=os.path.join("models", "L2CSNet_gaze360.pkl"))
    resources["gaze_engine"] = GazeEstimationEngine(gn, nca)
    try:
        yield
    finally:
        # Clean up the GazeEstimationEngine
        resources["gaze_engine"].shutdown()
        resources.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/load_profile")
def load_calibration_profile():
    pass


@app.post("/cal_point")
def add_calibration_point():
    pass


@app.post("/save_profile")
def save_current_profile():
    pass


@app.post("/predict")
async def predict_point_of_regard(
    file: UploadFile = File(...),  # Accept the uploaded image
):
    # Read the uploaded file's content as bytes
    image_bytes = await file.read()

    # Convert bytes to a NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the image array into an OpenCV image (BGR format)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gaze_engine = resources.get("gaze_engine")
    return gaze_engine.predict_gaze_position(frame)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
