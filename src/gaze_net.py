import warnings
from typing import Tuple

import torch
from l2cs import Pipeline

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch.load` with `weights_only=False`.*",
)


class GazeNet:
    """
    A wrapper for the gaze prediction pipeline.
    """

    def __init__(self, filepath: str):
        """
        Initialize the GazeNet pipeline.

        Args:
            filepath (str): Path to the model weights file.
        """
        self.gaze_pipeline = Pipeline(
            weights=filepath,
            arch="ResNet50",
            device=torch.device("cpu"),  # Use 'gpu' if available
        )

    @staticmethod
    def find_bounding_box_center(
        bounding_box: list, image_width: int
    ) -> Tuple[float, float]:
        """
        Calculate the center of the bounding box.

        Args:
            bounding_box (list): Coordinates of the bounding box [x_min, y_min, x_max, y_max].
            image_width (int): Width of the image.

        Returns:
            Tuple[float, float]: Center coordinates (x, y) of the bounding box.
        """
        x_min, y_min, x_max, y_max = bounding_box
        x_center = image_width - (x_min + x_max) / 2  # Invert x-axis.
        y_center = (y_min + y_max) / 2
        return x_center, y_center

    def predict_gaze_vector(
        self, image: torch.Tensor
    ) -> Tuple[float, float, float, float]:
        """
        Predict the gaze vector for a given image.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            Tuple[float, float, float, float]: Predicted gaze vector (x, y, pitch, yaw).
        """
        try:
            result = self.gaze_pipeline.step(image)
        except ValueError:
            print("No face detected.")
            return None, None, None, None

        bounding_box = result.bboxes[0]
        image_width = image.shape[1]
        x, y = self.find_bounding_box_center(bounding_box, image_width)

        return x, y, result.pitch[0], result.yaw[0]
