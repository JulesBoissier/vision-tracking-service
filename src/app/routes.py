from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile

from src.app.models import GazePredictionResponse, ProfileListResponse
from src.backend.calibration_agents import InterpolationAgent, LinearCalibrationAgent
from src.backend.calibration_profile_store import CalibrationProfileStore
from src.backend.gaze_predictor import GazePredictor
from src.backend.vision_tracking_engine import ProfileNotFoundError, VisionTrackingEngine
from src.settings import Settings

router = APIRouter()


def build_engine(settings: Settings) -> VisionTrackingEngine:
    """The production engine: the real L2CS model, the configured agent and profile store."""
    # CALIBRATION_AGENT=idw selects the previous inverse-distance interpolation, for A/B
    # comparisons on the benchmark. The default robust linear agent is far more accurate.
    agent = InterpolationAgent() if settings.calibration_agent == "idw" else LinearCalibrationAgent()
    return VisionTrackingEngine(
        GazePredictor(filepath=settings.model_path),
        agent,
        CalibrationProfileStore(db_url=settings.calibration_db_url),
    )


def create_app(
    settings: Optional[Settings] = None, engine: Optional[VisionTrackingEngine] = None
) -> FastAPI:
    """Build the app. Pass ``engine`` to serve a ready-made one (tests); otherwise one is
    built from ``settings`` (default: the environment) when the app starts.
    """
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or build_engine(settings)  # loads the model at startup
        try:
            yield
        finally:
            app.state.engine = None

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings
    app.include_router(router)
    return app


def get_engine(request: Request) -> VisionTrackingEngine:
    """FastAPI dependency: the engine of the app handling this request."""
    return request.app.state.engine


@router.get("/health")
def health_check():
    return {"status": "OK"}


@router.post("/save_profile")
def save_current_profile(name: str, vision_engine: VisionTrackingEngine = Depends(get_engine)):
    vision_engine.save_profile(name)
    return {"message": f"Profile '{name}' saved successfully."}


@router.get("/list_profiles")
def list_calibration_profiles(vision_engine: VisionTrackingEngine = Depends(get_engine)):
    profiles = vision_engine.list_profiles()
    return ProfileListResponse(profiles=profiles)


@router.post("/load_profile")
def load_calibration_profile(
    profile_id: int, vision_engine: VisionTrackingEngine = Depends(get_engine)
):
    try:
        vision_engine.load_profile(profile_id)
    except ProfileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No profile with id {profile_id}; the active calibration is unchanged.",
        )
    return {"message": "Profile loaded successfully."}


@router.post("/delete_profile")
def delete_calibration_profile(
    profile_id: int, vision_engine: VisionTrackingEngine = Depends(get_engine)
):
    vision_engine.delete_profile(profile_id)
    return {"message": "Profile deleted successfully."}


@router.post("/reset_profile")
def reset_calibration_profile(vision_engine: VisionTrackingEngine = Depends(get_engine)):
    vision_engine.reset_profile()
    return {"message": "Profile reset successfully."}


@router.post("/cal_point")
async def add_calibration_point(
    x: float = Form(...),  # Explicitly define x as a form field
    y: float = Form(...),  # Explicitly define y as a form field
    file: UploadFile = File(...),  # Accept the uploaded image
    vision_engine: VisionTrackingEngine = Depends(get_engine),
):
    # Read the uploaded file's content as bytes
    image_bytes = await file.read()

    # Convert bytes to a NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the image array into an OpenCV image (BGR format)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    added =vision_engine.run_single_calibration_step(x, y, frame)
    if not added:
        return {"message": "No face detected; calibration point skipped.", "added": False}
    return {
        "message": f"Calibration point added successfully with parameters x: {x}, y: {y}.",
        "added": True,
    }


@router.post("/predict")
async def predict_point_of_regard(
    # One or more JPEG frames, all as multipart fields named "file". Several frames (a
    # short burst) are combined into one prediction; a single frame works as always.
    file: List[UploadFile] = File(...),
    vision_engine: VisionTrackingEngine = Depends(get_engine),
):
    frames = []
    for upload in file:
        image_bytes = await upload.read()
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail=f"{upload.filename} is not a decodable image")
        frames.append(frame)

    predictions = vision_engine.predict_from_frames(frames)

    return GazePredictionResponse(prediction=predictions)


app = create_app()  # what main.py and the Dockerfile's `uvicorn main:app` serve
