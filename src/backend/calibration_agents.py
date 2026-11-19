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

    def __init__(self, position_interpolation_weight: float = 1):
        """
        Initialize the InterpolationAgent.
        """

        # Defines how much the head position is accounte for during interpolation
        self.position_interpolation_weight = position_interpolation_weight

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
            math.sqrt(
                (angle - calib_angle) ** 2
                + self.position_interpolation_weight * (position - calib_head) ** 2
            )
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


class NotCalibratedError(Exception):
    """Raised when a prediction is requested before any calibration point exists."""


class LinearCalibrationAgent(CalibrationAgent):
    """
    Robust linear calibration: screen position as a linear function of the two gaze angles.

    Design (measured on the attention-benchmark datasets, see docs/IMPROVEMENTS.md 3.1):
    - Gaze angles only. Head position from the face box is in pixels and jitters by a
      fraction of a pixel per frame. Mixed into one distance with angles in radians, it
      dominated the old interpolation. Head position also shifts when posture changes
      between calibration and use, so models that use it extrapolate badly.
    - Linear: for a flat screen, position is close to linear in the gaze angle over the
      angles a desk setup spans. Each screen axis is fitted from both angles, so a slightly
      rotated camera is absorbed by the fit.
    - Huber-weighted least squares: a calibration dot where the user blinked or looked away
      is down-weighted instead of skewing the whole fit.
    - Predictions are clamped to the area spanned by the calibration targets: the user is
      looking at the screen, and gaze outside the calibrated area can't be attributed to
      anything reliably.

    The calibration data structure is the same ``CalibrationMap`` as ``InterpolationAgent``,
    so saved profiles work with either agent. The fit is recomputed lazily whenever the
    map changes.
    """

    def __init__(self, huber_delta: float = 1.0, ridge: float = 1e-3, iterations: int = 20):
        self.huber_delta = huber_delta
        self.ridge = ridge
        self.iterations = iterations
        self.initialize_cal_map()

    @property
    def calibration_map(self) -> CalibrationMap:
        return self._calibration_map

    @calibration_map.setter
    def calibration_map(self, value: CalibrationMap):
        self._calibration_map = value
        self._model = None

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
        self.calibration_map.add_calibration_point(
            monitor_x, monitor_y, head_x, head_y, theta, phi
        )
        self._model = None

    def _fit(self):
        import numpy as np

        m = self.calibration_map
        if not m.theta_values:
            raise NotCalibratedError("no calibration points")
        X = np.column_stack([m.theta_values, m.phi_values]).astype(float)
        y = np.column_stack([m.monitor_x_values, m.monitor_y_values]).astype(float)

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-9] = 1.0  # a single point, or all points at one angle
        A = np.column_stack([np.ones(len(X)), (X - mean) / std])
        reg = self.ridge * np.eye(A.shape[1])
        reg[0, 0] = 0.0  # don't shrink the intercept

        weights = np.ones(len(X))
        for _ in range(self.iterations):
            Aw = A * weights[:, None]
            coef = np.linalg.solve(A.T @ Aw + reg, Aw.T @ y)
            residual = np.hypot(*(A @ coef - y).T)
            scale = np.median(residual) / 0.6745  # robust estimate of residual spread
            if scale < 1e-9:
                break
            u = residual / (self.huber_delta * scale)
            weights = 1.0 / np.maximum(u, 1.0)  # Huber: 1 inside delta, 1/u beyond

        self._model = (mean, std, coef, y.min(axis=0), y.max(axis=0))

    def calculate_point_of_regard(
        self, head_x: float, head_y: float, theta: float, phi: float
    ) -> Tuple[float, float]:
        """Screen coordinates for the given gaze angles. Head position is accepted but unused."""
        if theta is None or phi is None:
            raise TypeError("no gaze angles (no face detected)")
        if self._model is None:
            self._fit()
        mean, std, coef, lo, hi = self._model
        features = [1.0, (theta - mean[0]) / std[0], (phi - mean[1]) / std[1]]
        x = sum(f * c for f, c in zip(features, coef[:, 0]))
        y = sum(f * c for f, c in zip(features, coef[:, 1]))
        return (
            float(min(max(x, lo[0]), hi[0])),
            float(min(max(y, lo[1]), hi[1])),
        )
