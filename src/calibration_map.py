from dataclasses import dataclass, field
from typing import List


@dataclass
class CalibrationMap:
    """
    A data class to store calibration points.
    """

    x_values: List[float] = field(default_factory=list)
    y_values: List[float] = field(default_factory=list)
    theta_values: List[float] = field(default_factory=list)
    phi_values: List[float] = field(default_factory=list)

    def add_calibration_point(
        self, x_value: float, y_value: float, theta_value: float, phi_value: float
    ):
        """
        Add a calibration point to the map.

        Args:
            x_value (float): X coordinate of the screen.
            y_value (float): Y coordinate of the screen.
            theta_value (float): Gaze angle in horizontal direction.
            phi_value (float): Gaze angle in vertical direction.
        """
        self.x_values.append(x_value)
        self.y_values.append(y_value)
        self.theta_values.append(theta_value)
        self.phi_values.append(phi_value)
