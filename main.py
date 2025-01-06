from src.calibration_agent import CalibrationAgent
from src.gaze_net import GazeNet
from src.gaze_estimation_engine import GazeEstimationEngine

if __name__ == '__main__':

    ca = CalibrationAgent('yo')

    print(ca.calculate_point_of_regard(-0.3, 0.2, 0, 0))