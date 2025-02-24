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
        self, head_x: float, head_y: float, theta: float, phi: float
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


class InterpolationAgent(CalibrationAgent):
    """
    An interpolation based implementation of a CalibrationAgent assuming static head position.
    """

    def __init__(self):
        """
        Initialize the InterpolationAgent.
        """
        self.initialize_cal_map()

    def initialize_cal_map(self):
        self.calibration_map = CalibrationMap()

    def calibration_step(
        self,
        monitor_x: float,
        monitor_y: float,
        head_x: float,
        head_y: float,
        theta: float,
        phi: float,
    ):
        """
        Add a calibration point during the calibration process.

        Args:
            x (float): X coordinate on the screen.
            y (float): Y coordinate on the screen.
            theta (float): Horizontal gaze angle.
            phi (float): Vertical gaze angle.
        """
        self.calibration_map.add_calibration_point(
            monitor_x, monitor_y, head_x, head_y, theta, phi
        )

    def _interpolate(
        self,
        position: float,
        angle: float,
        calibration_monitor_coordinates: List[float],
        calibration_head_coordinates: List[float],
        calibration_angles: List[float],
    ) -> float:
        """
        Interpolate the screen coordinate based on both head position and gaze angle.

        Args:
            position (float): The current head position.
            angle (float): The current gaze angle.
            calibration_monitor_coordinates (List[float]): Corresponding screen coordinates.
            calibration_head_coordinates (List[float]): Corresponding head positions.
            calibration_angles (List[float]): Corresponding gaze angles.

        Returns:
            float: Interpolated screen coordinate.
        """
        epsilon = 1e-6
        # Calculate the combined Euclidean distance in the (position, angle) space.
        distances = [
            math.sqrt((angle - calib_angle) ** 2 + (position - calib_head) ** 2)
            for calib_angle, calib_head in zip(
                calibration_angles, calibration_head_coordinates
            )
        ]
        # Compute weights inversely proportional to the combined distance.
        weights = [1 / (d + epsilon) for d in distances]
        # Weighted sum of monitor coordinates.
        numerator = sum(
            w * coord for w, coord in zip(weights, calibration_monitor_coordinates)
        )
        return numerator / sum(weights)

    def calculate_point_of_regard(
        self, head_x: float, head_y: float, theta: float, phi: float
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
            head_x,
            theta,
            self.calibration_map.monitor_x_values,
            self.calibration_map.head_x_values,
            self.calibration_map.theta_values,
        )
        y_screen = self._interpolate(
            head_y,
            phi,
            self.calibration_map.monitor_y_values,
            self.calibration_map.head_y_values,
            self.calibration_map.phi_values,
        )
        return x_screen, y_screen
