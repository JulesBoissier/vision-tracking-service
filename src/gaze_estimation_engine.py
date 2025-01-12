from typing import List, Tuple
import numpy as np
from src.gaze_net import GazeNet
from src.calibration_agents import CalibrationAgent


class GazeEstimationEngine:
    
    def __init__(self, gaze_net : GazeNet, cal_agent : CalibrationAgent):
        self.gaze_net = gaze_net
        self.cal_agent = cal_agent

    def run_calibration_steps(self, calibration_data : List[Tuple[int, int, np.ndarray]]):

        for calibration_point in calibration_data:
            _, _, theta, phi = self.gaze_net.predict_gaze_vector(calibration_point[2])
            self.cal_agent.calibration_step(calibration_point[0], calibration_point[1], theta, phi)
            

    def predict_gaze_position(self, image):
        _, _, theta, phi = self.gaze_net.predict_gaze_vector(image)
        screen_x, screen_y = self.cal_agent.calculate_point_of_regard(theta, phi)
        return screen_x, screen_y