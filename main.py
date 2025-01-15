import uvicorn
from fastapi import FastAPI

from src.calibration_agents import NaiveCalibrationAgent
from src.gaze_estimation_engine import GazeEstimationEngine
from src.gaze_net import GazeNet

app = FastAPI()


@app.get("/load_profile")
def load_calibration_profile():
    pass


@app.post("/cal_point")
def add_calibration_point():
    pass


@app.post("/save_profile")
def save_current_profile():
    pass


@app.get("/predict")
def predict_point_of_regard():
    pass


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
