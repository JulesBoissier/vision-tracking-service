from src.gaze_net import GazeNet
from src.calibration_agent import CalibrationAgent


class GazeEstimationEngine:
    
    def __init__(self, gaze_net : GazeNet, cal_agent : CalibrationAgent):
        self.gaze_net = gaze_net
        self.cal_agent = cal_agent

    def predict_gaze_position(self, image):
        x, y, theta, phi = self.gaze_net.predict_gaze_vector(image)
        screen_n, screen_x, screen_y = self.calculate_point_of_regard(x, y, theta, phi)
        return screen_n, screen_x, screen_y