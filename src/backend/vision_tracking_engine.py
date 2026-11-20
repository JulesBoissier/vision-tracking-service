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

    def run_single_calibration_step(self, screen_x: float, screen_y: float, frame, implicit: bool = False) -> bool:
        """
        Add one calibration point: the user was looking at (screen_x, screen_y) in ``frame``.

        ``frame`` may be one image or a burst (list); a burst's gaze features are averaged.
        ``implicit=True`` marks evidence gathered during use (a click) rather than an
        explicit calibration dot: it goes to the agent's per-session implicit pool, isn't
        saved in the profile, and is refused (False) by agents that don't support it.

        Frames where no face is detected are skipped, because a point without gaze features
        would make every later prediction fail.

        Returns:
            bool: True if the point was added.
        """
        features = self._average_features(frame if isinstance(frame, list) else [frame])
        if features is None:
            return False
        head_x, head_y, theta, phi, face_width = features
        with self._lock:
            if implicit:
                if not self.cal_agent.supports_implicit:
                    return False
                self.cal_agent.add_implicit_point(
                    screen_x, screen_y, head_x, head_y, theta, phi, face_width=face_width
                )
            else:
                self.cal_agent.calibration_step(
                    screen_x, screen_y, head_x, head_y, theta, phi, face_width=face_width
                )
        return True

    def _average_features(self, frames):
        """Mean ``(head_x, head_y, theta, phi, face_width)`` over frames with a face, or None."""
        features = [self.gaze_predictor.predict_gaze_features(f) for f in frames]
        features = [f for f in features if f[2] is not None and f[3] is not None]
        if not features:
            return None
        head_x, head_y, theta, phi = (float(np.mean([f[i] for f in features])) for i in range(4))
        widths = [f[4] for f in features if f[4]]
        return head_x, head_y, theta, phi, (float(np.mean(widths)) if widths else None)

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
        """Predict the on-screen gaze position for one frame; ``(None, None)`` if impossible."""
        return self.predict_from_frames([image])

    def predict_from_frames(self, frames: List[np.ndarray]) -> Tuple[float, float]:
        """Predict one on-screen gaze position from a short burst of frames.

        Gaze features (head position and angles) are averaged over the frames where a face
        was found, then mapped through the calibration once. Averaging consecutive frames
        reduces frame-to-frame jitter: on the benchmark, 3 frames cut the median error on
        three monitors by ~6% and raised right-monitor attribution from 80% to 83%.
        Averaging features beat taking the median of per-frame predictions. A single frame
        behaves exactly like before.

        Returns ``(None, None)`` when no frame has a face or nothing is calibrated.
        """
        features = self._average_features(frames)
        if features is None:
            return None, None  # no face detected in any frame
        head_x, head_y, theta, phi, face_width = features

        try:
            with self._lock:
                screen_x, screen_y = self.cal_agent.calculate_point_of_regard(
                    head_x, head_y, theta, phi, face_width=face_width
                )
        except (NotCalibratedError, ZeroDivisionError):
            # No calibration points yet (ZeroDivisionError comes from InterpolationAgent).
            return None, None
        return screen_x, screen_y
