from dataclasses import dataclass, field
from typing import List


@dataclass
class CalibrationMap:
    """
    A data class to store calibration points.
    """

    monitor_x_values: List[float] = field(default_factory=list)
    monitor_y_values: List[float] = field(default_factory=list)
    head_x_values: List[float] = field(default_factory=list)
    head_y_values: List[float] = field(default_factory=list)
    theta_values: List[float] = field(default_factory=list)
    phi_values: List[float] = field(default_factory=list)

    def add_calibration_point(
        self,
        monitor_x_value: float,
        monitor_y_value: float,
        head_x_value: float,
        head_y_value: float,
        theta_value: float,
        phi_value: float,
    ):
        """
        Add a calibration point to the map.

        Args:
            x_value (float): X coordinate of the screen.
            y_value (float): Y coordinate of the screen.
            theta_value (float): Gaze angle in horizontal direction.
            phi_value (float): Gaze angle in vertical direction.
        """
        print("ADDING CAL POINT:")
        print(monitor_x_value)
        print(monitor_y_value)
        print(head_x_value)
        print(head_y_value)

        self.monitor_x_values.append(monitor_x_value)
        self.monitor_y_values.append(monitor_y_value)
        self.head_x_values.append(head_x_value)
        self.head_y_values.append(head_y_value)
        self.theta_values.append(theta_value)
        self.phi_values.append(phi_value)
