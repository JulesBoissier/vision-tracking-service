from src.gaze_net import GazeNet
from src.calibration_agent import CalibrationAgent


class GazeEstimationEngine:
    
    def __init__(self, gaze_net : GazeNet, cal_agent : CalibrationAgent):
        self.gaze_net = gaze_net
        self.cal_agent = cal_agent

    def run_calibration_steps(self, cal_steps : int = 9):

        t = 0

        for i in range(cal_steps):
            
            x_target, y_target = 0, 0

            image = "uhoh"  #! How do we get an image from inside the loop?

            x, y, theta, phi = self.gaze_net.predict_gaze_vector(image)
            
            t += self.cal_agent.calibration_step(x, y, theta, phi, x_target, y_target)
        
        t /= cal_steps

        self.cal_agent.t = t  #! Use a setter here as opposed to accessing attribute directly.



    def predict_gaze_position(self, image):
        x, y, theta, phi = self.gaze_net.predict_gaze_vector(image)
        screen_n, screen_x, screen_y = self.calculate_point_of_regard(x, y, theta, phi)
        return screen_n, screen_x, screen_y