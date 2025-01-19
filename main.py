import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

from src.calibration_agents import NaiveCalibrationAgent
from src.calibration_data_store import CalibrationDataStore
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
    cds = CalibrationDataStore()
    gn = GazeNet(filepath=os.path.join("models", "L2CSNet_gaze360.pkl"))
    resources["gaze_engine"] = GazeEstimationEngine(gn, nca, cds)
    try:
        yield
    finally:
        # Clean up the GazeEstimationEngine
        resources["gaze_engine"].shutdown()
        resources.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/save_profile")
def save_current_profile():
    gaze_engine = resources.get("gaze_engine")
    gaze_engine.save_profile("Test")


@app.get("/list_profiles")
def list_calibration_profiles():
    gaze_engine = resources.get("gaze_engine")
    return gaze_engine.list_profiles()


@app.post("/load_profile")
def load_calibration_profile(profile_id: int):
    gaze_engine = resources.get("gaze_engine")
    gaze_engine.load_profile(profile_id)


@app.post("/cal_point")
async def add_calibration_point(
    x: float = Form(...),  # Explicitly define x as a form field
    y: float = Form(...),  # Explicitly define y as a form field
    file: UploadFile = File(...),  # Accept the uploaded image
):
    # Read the uploaded file's content as bytes
    image_bytes = await file.read()

    # Convert bytes to a NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the image array into an OpenCV image (BGR format)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gaze_engine = resources.get("gaze_engine")

    gaze_engine.run_single_calibration_step(x, y, frame)
    return {
        "message": f"Calibration point added successfully with parameters x: {x}, y: {y}."
    }


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
    uvicorn.run(app, host="127.0.0.1", port=8001)
