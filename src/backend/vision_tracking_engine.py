from typing import List, Tuple

import numpy as np

from src.backend.calibration_agents import CalibrationAgent
from src.backend.calibration_profile_store import CalibrationProfileStore
from src.backend.gaze_predictor import GazePredictor


class VisionTrackingEngine:
    """
    A class responsible for estimating gaze positions using GazePredictor and CalibrationAgent.
    """

    def __init__(
        self,
        gaze_predictor: GazePredictor,
        cal_agent: CalibrationAgent,
        cps=CalibrationProfileStore,
    ):
        """
        Initialize the VisionTrackingEngine.

        Args:
            gaze_predictor (GazePredictor): An instance of GazePredictor for predicting gaze vectors.
            cal_agent (CalibrationAgent): An instance of CalibrationAgent for calibration tasks.
            cps (CalibrationProfileStore): An instance of CalibrationProfileStore for storing and retrieving calibration profiles.
        """
        self.gaze_predictor = gaze_predictor
        self.cal_agent = cal_agent
        self.cps = cps

    def save_profile(self, name):
        self.cps.save_profile(name, self.cal_agent.calibration_map)

    def load_profile(self, id):
        self.cal_agent.calibration_map = self.cps.load_profile(id)

    def list_profiles(self):
        return self.cps.list_profiles()

    def delete_profile(self, id):
        self.cps.delete_profile(id)

    def run_single_calibration_step(
        self, head_x: float, head_y: float, frame: np.ndarray
    ):
        """
        Perform a single calibration step using GazePredictor and CalibrationAgent.

        Args:
            x (float): X coordinate on the screen.
            y (float): Y coordinate on the screen.
            frame (np.ndarray): The input image for gaze prediction.
        """
        monitor_x, monitor_y, theta, phi = self.gaze_predictor.predict_gaze_vector(
            frame
        )
        self.cal_agent.calibration_step(
            head_x, head_y, monitor_x, monitor_y, theta, phi
        )

    def run_calibration_steps(
        self, calibration_data: List[Tuple[int, int, np.ndarray]]
    ):
        """
        Perform calibration steps for provided data.

        Args:
            calibration_data (List[Tuple[int, int, np.ndarray]]): List of tuples containing
            x, y screen coordinates and corresponding image data.
        """
        for calibration_point in calibration_data:
            try:
                self.run_single_calibration_step(
                    calibration_point[0], calibration_point[1], calibration_point[2]
                )
            except Exception as e:
                print(f"Calibration step failed for point {calibration_point}: {e}")

    def predict_gaze_position(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Predict the gaze position on the screen for a given image.

        Args:
            image (np.ndarray): The input image for gaze prediction.

        Returns:
            Tuple[float, float]: The predicted screen coordinates (x, y).
        """
        _, _, theta, phi = self.gaze_predictor.predict_gaze_vector(image)

        try:
            screen_x, screen_y = self.cal_agent.calculate_point_of_regard(theta, phi)

        except ZeroDivisionError:
            print("Calibration profile is empty.")
            screen_x, screen_y = None, None
        except TypeError:
            print("No face detected.")
            screen_x, screen_y = None, None
        return screen_x, screen_y
