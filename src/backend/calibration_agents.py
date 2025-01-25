import math
from abc import ABC, abstractmethod
from typing import List, Tuple

from src.backend.calibration_map import CalibrationMap


class CalibrationAgent(ABC):
    """
    Abstract base class for calibration agents.
    """

    @abstractmethod
    def calculate_point_of_regard(
        self, theta: float, phi: float
    ) -> Tuple[float, float]:
        """
        Calculate the point of regard on the screen.

        Args:
            theta (float): The horizontal gaze angle.
            phi (float): The vertical gaze angle.

        Returns:
            Tuple[float, float]: Screen coordinates (x, y).
        """
        pass


class NaiveCalibrationAgent(CalibrationAgent):
    """
    A naive implementation of a CalibrationAgent assuming static head position.
    """

    def __init__(self):
        """
        Initialize the NaiveCalibrationAgent.
        """
        self.calibration_map = CalibrationMap()

    def calibration_step(self, x: float, y: float, theta: float, phi: float):
        """
        Add a calibration point during the calibration process.

        Args:
            x (float): X coordinate on the screen.
            y (float): Y coordinate on the screen.
            theta (float): Horizontal gaze angle.
            phi (float): Vertical gaze angle.
        """
        self.calibration_map.add_calibration_point(x, y, theta, phi)

    def _interpolate(
        self,
        angle: float,
        calibration_coordinates: List[float],
        calibration_angles: List[float],
    ) -> float:
        """
        Interpolate the screen coordinate based on calibration data.

        Args:
            angle (float): The gaze angle to interpolate.
            calibration_coordinates (List[float]): Corresponding screen coordinates.
            calibration_angles (List[float]): Corresponding gaze angles.

        Returns:
            float: Interpolated screen coordinate.
        """
        epsilon = 1e-6
        distances = [
            math.sqrt((angle - calib_angle) ** 2) for calib_angle in calibration_angles
        ]
        weights = [1 / (distance + epsilon) for distance in distances]
        numerator = sum(w * coord for w, coord in zip(weights, calibration_coordinates))
        return numerator / sum(weights)

    def calculate_point_of_regard(
        self, theta: float, phi: float
    ) -> Tuple[float, float]:
        """
        Calculate the screen coordinates for a given gaze angle.

        Args:
            theta (float): Horizontal gaze angle.
            phi (float): Vertical gaze angle.

        Returns:
            Tuple[float, float]: Screen coordinates (x, y).
        """
        x_screen = self._interpolate(
            theta, self.calibration_map.x_values, self.calibration_map.theta_values
        )
        y_screen = self._interpolate(
            phi, self.calibration_map.y_values, self.calibration_map.phi_values
        )
        return x_screen, y_screen
