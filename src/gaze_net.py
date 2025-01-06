def mock_gaze_net(image):
    
    import random

    x = random.randint(0, 100) / 100
    y = random.randint(0, 100) / 100
    theta = random.randint(0, 180)
    phi = random.randint(0, 180)

    return x, y, theta, phi

class GazeNet:  #? A wrapper to multiple nets?
    def __init__(self, filepath):
        pass

    def predict_gaze_vector(self, image):
        return mock_gaze_net(image)