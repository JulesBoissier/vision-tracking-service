from typing import List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
import numpy as np

class CalibrationAgent(ABC):
    @abstractmethod
    def create_profile(self):
        pass

    @abstractmethod
    def load_profile(self):
        pass
    
    @abstractmethod
    def calculate_point_of_regard(self):
        pass

@dataclass
class CalibrationMap:
    """
    A map for storing calibration points.
    """
    x_values: List[float] = field(default_factory=list)
    y_values: List[float] = field(default_factory=list)
    theta_values: List[float] = field(default_factory=list)
    phi_values: List[float] = field(default_factory=list)
    
    def add_calibration_point(self, x_value, y_value, theta_value, phi_value):
        self.x_values.append(x_value)
        self.y_values.append(y_value)
        self.theta_values.append(theta_value)
        self.phi_values.append(phi_value)

        print("Added calibration point.")

        
class NaiveCalibrationAgent(CalibrationAgent):
    """
    Assuming a static head-position, simply checking which
    monitor square the gaze vector hits.
    """

    def __init__(self, db_path : str):

        self.db_path = db_path
        self.calibration_map = CalibrationMap()
        # self.screen_dimensions = (0, 0)  # Must be given pixels 

    def create_profile(self):
        pass
    
    def load_profile(self):
        pass
    
    def calibration_step(self, x, y, theta, phi):
        self.calibration_map.add_calibration_point(x, y, theta, phi)


    def _interpolate(self, angle, calibration_coordinates, calibration_angles):
        epsilon = 10**-6
        distances = [math.sqrt((angle - calibration_angle)**2) for calibration_angle in calibration_angles]
        weights = [1 / (distance + epsilon) for distance in distances]  # Could be made into separate method  for more interpolation methods.
        numerator = [weight * calibration_coordinate for weight, calibration_coordinate in zip(weights, calibration_coordinates)]
        return sum(numerator) / sum(weights)
    
    def calculate_point_of_regard(self, theta, phi):
        x_screen = self._interpolate(theta, self.calibration_map.x_values, self.calibration_map.theta_values)
        y_screen = self._interpolate(phi, self.calibration_map.y_values, self.calibration_map.phi_values)
        return x_screen, y_screen


class PreciseCalibrationAgent(CalibrationAgent):

    def __init__(self, db_path):

        # TODO: Need eye position in screen space, not camera space.

        self.t = 1
        self.screen_params = {
            'X_screen': -0.3,  # Top-left corner (X) in meters
            'Y_screen': 0.2,  # Top-left corner (Y) in meters
            'W_screen': 0.6,  # Screen width in meters
            'H_screen': 0.4,  # Screen height in meters
            'W_pixels': 1920,  # Screen width in pixels
            'H_pixels': 1080   # Screen height in pixels
        }

    def create_profile(self):
        pass
    
    def _save_profile(self):
        pass

    def load_profile(self):
        pass

    def calibration_step(self, x, y, theta, phi, x_target, y_target):
        horizontal_t = (x - x_target) / (math.cos(phi) * math.cos(theta))
        vertical_t = (y - y_target) / (math.cos(phi) * math.sin(theta))
        return (horizontal_t + vertical_t) / 2
    
    

    def calculate_point_of_regard(self, x, y, theta, phi):
        # Calculate gaze vector
        gaze_vector = np.array([
            math.cos(phi) * math.cos(theta),
            math.cos(phi) * math.sin(theta),
            math.sin(phi)
        ])

        # Handle division by zero for t
        if phi == 0:
            por_x = x
            por_y = y
        else:
            # Calculate screen intersection point (X, Y)
            por_x = x + self.t * gaze_vector[0]
            por_y = y + self.t * gaze_vector[1]
        
        # Map physical coordinates to screen pixel coordinates
        screen_x = ((por_x - self.screen_params['X_screen']) / self.screen_params['W_screen']) * self.screen_params['W_pixels']
        screen_y = -((por_y - self.screen_params['Y_screen']) / self.screen_params['H_screen']) * self.screen_params['H_pixels']
        return screen_x, screen_y
