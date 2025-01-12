from typing import List, Tuple

import numpy as np

from src.calibration_agents import CalibrationAgent
from src.gaze_net import GazeNet


class GazeEstimationEngine:
    """
    A class responsible for estimating gaze positions using GazeNet and CalibrationAgent.
    """

    def __init__(self, gaze_net: GazeNet, cal_agent: CalibrationAgent):
        """
        Initialize the GazeEstimationEngine.

        Args:
            gaze_net (GazeNet): An instance of GazeNet for predicting gaze vectors.
            cal_agent (CalibrationAgent): An instance of CalibrationAgent for calibration tasks.
        """
        self.gaze_net = gaze_net
        self.cal_agent = cal_agent

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
                _, _, theta, phi = self.gaze_net.predict_gaze_vector(
                    calibration_point[2]
                )
                self.cal_agent.calibration_step(
                    calibration_point[0], calibration_point[1], theta, phi
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
        _, _, theta, phi = self.gaze_net.predict_gaze_vector(image)
        screen_x, screen_y = self.cal_agent.calculate_point_of_regard(theta, phi)
        return screen_x, screen_y
