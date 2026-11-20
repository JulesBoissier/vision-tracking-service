import threading
from typing import List, Tuple

import numpy as np

from src.backend.calibration_agents import CalibrationAgent, NotCalibratedError
from src.backend.calibration_profile_store import CalibrationProfileStore
from src.backend.gaze_predictor import GazePredictor


class ProfileNotFoundError(KeyError):
    """Raised when loading a profile id that doesn't exist."""


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
        # Serialises everything that reads or replaces the calibration. FastAPI runs sync
        # routes (load/reset/save) in a thread pool, concurrently with calibration and
        # prediction. Gaze inference itself runs outside the lock.
        self._lock = threading.RLock()

    def save_profile(self, name):
        with self._lock:
            self.cps.save_profile(name, self.cal_agent.calibration_map)

    def load_profile(self, id):
        """Make a saved profile the active calibration. Unknown ids leave it unchanged."""
        calibration_map = self.cps.load_profile(id)
        if calibration_map is None:
            raise ProfileNotFoundError(id)
        with self._lock:
            self.cal_agent.calibration_map = calibration_map

    def reset_profile(self):
        """Discard the active calibration."""
        with self._lock:
            self.cal_agent.initialize_cal_map()

    def list_profiles(self):
        return self.cps.list_profiles()

    def delete_profile(self, id):
        self.cps.delete_profile(id)

    def run_single_calibration_step(
        self, screen_x: float, screen_y: float, frame: np.ndarray
    ) -> bool:
        """
        Add one calibration point: the user was looking at (screen_x, screen_y) in ``frame``.

        Frames where no face is detected are skipped, because a point without gaze features
        would make every later prediction fail.

        Returns:
            bool: True if the point was added, False if no face was found.
        """
        head_x, head_y, theta, phi = self.gaze_predictor.predict_gaze_vector(frame)
        if theta is None or phi is None:
            return False
        with self._lock:
            self.cal_agent.calibration_step(screen_x, screen_y, head_x, head_y, theta, phi)
        return True

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
        head_x, head_y, theta, phi = self.gaze_predictor.predict_gaze_vector(image)
        if theta is None or phi is None:
            return None, None  # no face detected

        try:
            with self._lock:
                screen_x, screen_y = self.cal_agent.calculate_point_of_regard(
                    head_x, head_y, theta, phi
                )
        except (NotCalibratedError, ZeroDivisionError):
            # No calibration points yet (ZeroDivisionError comes from InterpolationAgent).
            return None, None
        return screen_x, screen_y
