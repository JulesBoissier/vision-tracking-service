from dataclasses import dataclass, field
from typing import List, Optional


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
    # Face width in pixels per point (a distance proxy). Profiles saved before this field
    # existed don't have it; use ``face_widths()``, which pads them with None.
    face_width_values: List[Optional[float]] = field(default_factory=list)

    def face_widths(self) -> List[Optional[float]]:
        """One face width per calibration point, ``None`` where it wasn't recorded."""
        widths = self.__dict__.setdefault("face_width_values", [])
        missing = len(self.theta_values) - len(widths)
        if missing > 0:  # points recorded before face widths were stored
            widths[:0] = [None] * missing
        return widths

    def add_calibration_point(
        self,
        monitor_x_value: float,
        monitor_y_value: float,
        head_x_value: float,
        head_y_value: float,
        theta_value: float,
        phi_value: float,
        face_width_value: Optional[float] = None,
    ):
        """
        Add a calibration point to the map.

        Args:
            x_value (float): X coordinate of the screen.
            y_value (float): Y coordinate of the screen.
            theta_value (float): Gaze angle in horizontal direction.
            phi_value (float): Gaze angle in vertical direction.
        """
        widths = self.face_widths()  # before appending: pads points recorded without widths
        self.monitor_x_values.append(monitor_x_value)
        self.monitor_y_values.append(monitor_y_value)
        self.head_x_values.append(head_x_value)
        self.head_y_values.append(head_y_value)
        self.theta_values.append(theta_value)
        self.phi_values.append(phi_value)
        widths.append(face_width_value)
