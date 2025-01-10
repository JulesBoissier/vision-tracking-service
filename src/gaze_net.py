import warnings

import torch
from l2cs import Pipeline

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch.load` with `weights_only=False`.*",
)

class GazeNet: 
    def __init__(self, filepath):
        self.gaze_pipeline = Pipeline(
        weights=filepath,
        arch='ResNet50',
        device=torch.device('cpu')  # or 'gpu'
    )
    @staticmethod
    def find_boundind_box_center(bounding_box, image_width):

        x_min, y_min, x_max, y_max = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
        
        x_center = image_width - (x_min + x_max) / 2  # Inverting x axis.
        y_center = (y_min + y_max) / 2

        return x_center, y_center  # Inverting y axis.

    def predict_gaze_vector(self, image):
        try:
            result = self.gaze_pipeline.step(image)  # With image being _, image = cap.read()
        except ValueError:
            print("No face detected.")
            return None, None, None, None

        bounding_box = result.bboxes[0]
        image_width = image.shape[1]
        x, y = self.find_boundind_box_center(bounding_box, image_width)

        return result.pitch, result.yaw, x, y
