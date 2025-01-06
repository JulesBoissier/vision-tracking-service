import math
import numpy as np


class CalibrationAgent:

    def __init__(self, db_path):
        self.z_eye = 0

    def create_profile(self):
        pass
    
    def _save_profile(self):
        pass

    def load_profile(self):
        pass

    def calculate_point_of_regard(self, x, y , theta, phi):
        # Calculate gaze vector
        gaze_vector = np.array([
            math.cos(phi) * math.cos(theta),
            math.cos(phi) * math.sin(theta),
            math.sin(phi)
        ])
        print(gaze_vector)
        # Screen and eye depths
        self.z_screen = 0
        self.z_eye = 1

        screen_params = {
            'X_screen': 0,  # Screen center (X) in meters
            'Y_screen': 0,  # Screen center (Y) in meters
            'W_screen': 0.6,  # Screen width in meters
            'H_screen': 0.4,  # Screen height in meters
            'W_pixels': 1920,  # Screen width in pixels
            'H_pixels': 1080   # Screen height in pixels
        }

        # Handle division by zero for t
        if phi == 0:
            por_x = x
            por_y = y
        else:
            # Calculate intersection (Line of Sight with Screen Plane)
            t = (self.z_screen - self.z_eye) / gaze_vector[2]
            # Calculate screen intersection point (X, Y)
            por_x = x + t * gaze_vector[0]
            por_y = y + t * gaze_vector[1]
        
        print("PORs:", por_x, por_y)
        
        # Extract screen parameters
        X_screen = screen_params['X_screen']  # Screen center X (meters)
        Y_screen = screen_params['Y_screen']  # Screen center Y (meters)
        W_screen = screen_params['W_screen']  # Screen width (meters)
        H_screen = screen_params['H_screen']  # Screen height (meters)
        W_pixels = screen_params['W_pixels']  # Screen width (pixels)
        H_pixels = screen_params['H_pixels']  # Screen height (pixels)

        # Map physical coordinates to screen pixel coordinates
        screen_x = ((por_x - X_screen) / W_screen) * W_pixels + W_pixels / 2
        screen_y = -((por_y - Y_screen) / H_screen) * H_pixels + H_pixels / 2

        return screen_x, screen_y



    def calibrate_point_of_regard(self, uncal_por, kappa):  #Should be self.kappa
        #* Differ with uncal_por by a minimal angle kappa
        pass #! Visual Axis instead of Optical Axis