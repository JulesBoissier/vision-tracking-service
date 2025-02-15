import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

from src.app.models import GazePredictionResponse, ProfileListResponse
from src.backend.calibration_agents import InterpolationAgent
from src.backend.calibration_profile_store import CalibrationProfileStore
from src.backend.gaze_predictor import GazePredictor
from src.backend.vision_tracking_engine import VisionTrackingEngine

# Dictionary to hold the VisionTrackingEngine instance
resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the VisionTrackingEngine

    ia = InterpolationAgent()
    cps = CalibrationProfileStore(db_url="sqlite:///calibration.db")
    gp = GazePredictor(filepath=os.path.join("models", "L2CSNet_gaze360.pkl"))
    resources["vision_engine"] = VisionTrackingEngine(gp, ia, cps)
    try:
        yield
    finally:
        # Clean up the VisionTrackingEngine
        resources.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/save_profile")
def save_current_profile(name: str):
    vision_engine = resources.get("vision_engine")
    vision_engine.save_profile(name)
    return {"message": f"Profile '{name}' saved successfully."}


@app.get("/list_profiles")
def list_calibration_profiles():
    vision_engine = resources.get("vision_engine")
    profiles = vision_engine.list_profiles()
    return ProfileListResponse(profiles=profiles)


@app.post("/load_profile")
def load_calibration_profile(profile_id: int):
    vision_engine = resources.get("vision_engine")
    vision_engine.load_profile(profile_id)
    return {"message": "Profile loaded successfully."}


@app.post("/delete_profile")
def delete_calibration_profile(profile_id: int):
    vision_engine = resources.get("vision_engine")
    vision_engine.delete_profile(profile_id)
    return {"message": "Profile deleted successfully."}


@app.post("/reset_profile")
def reset_calibration_profile():
    vision_engine = resources.get("vision_engine")
    vision_engine.cal_agent.initialize_cal_map()
    return {"message": "Profile reset successfully."}


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

    vision_engine = resources.get("vision_engine")

    vision_engine.run_single_calibration_step(x, y, frame)
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

    vision_engine = resources.get("vision_engine")
    predictions = vision_engine.predict_gaze_position(frame)

    return GazePredictionResponse(prediction=predictions)
