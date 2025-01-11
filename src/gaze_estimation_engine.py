from typing import List, Tuple, Any
from src.gaze_net import GazeNet
from src.calibration_agents import CalibrationAgent


class GazeEstimationEngine:
    
    def __init__(self, gaze_net : GazeNet, cal_agent : CalibrationAgent):
        self.gaze_net = gaze_net
        self.cal_agent = cal_agent

    def run_calibration_steps(self, calibration_data : List[Tuple[int, int, Any]]):

        t = 0

        for calibration_data in calibration_data:
            
            x_target, y_target = calibration_data[0], calibration_data[1]

            image = calibration_data[2]

            x, y, theta, phi = self.gaze_net.predict_gaze_vector(image)
            
            t += self.cal_agent.calibration_step(x, y, theta, phi, x_target, y_target)
        
        t /= len(calibration_data)

        self.cal_agent.t = t  #! Use a setter here as opposed to accessing attribute directly.



    def predict_gaze_position(self, image):
        x, y, theta, phi = self.gaze_net.predict_gaze_vector(image)
        screen_n, screen_x, screen_y = self.cal_agent.calculate_point_of_regard(x, y, theta, phi)
        return screen_n, screen_x, screen_y