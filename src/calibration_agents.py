from abc import ABC, abstractmethod
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



class PreciseCalibrationAgent(CalibrationAgent):

    def __init__(self, db_path):

        raise NotImplementedError
    
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
