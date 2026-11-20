import warnings
from typing import Tuple

import torch
from l2cs import Pipeline

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch.load` with `weights_only=False`.*",
)


class GazePredictor:
    """
    A wrapper for the gaze prediction pipeline.
    """

    def __init__(self, filepath: str):
        """
        Initialize the GazePredictor pipeline.

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

    def predict_gaze_features(self, image):
        """Gaze features for one image: ``(head_x, head_y, pitch, yaw, face_width)``.

        Head position is the face box centre (x mirrored), face width the face box width in
        pixels: a proxy for distance to the camera (closer means wider). All ``None`` when
        no face is detected.
        """
        try:
            result = self.gaze_pipeline.step(image)
        except ValueError:
            return None, None, None, None, None

        bounding_box = result.bboxes[0]
        image_width = image.shape[1]
        x, y = self.find_bounding_box_center(bounding_box, image_width)
        face_width = float(bounding_box[2] - bounding_box[0])
        return x, y, result.pitch[0], result.yaw[0], face_width

    def predict_gaze_vector(
        self, image: torch.Tensor
    ) -> Tuple[float, float, float, float]:
        """``(head_x, head_y, pitch, yaw)`` for one image; four ``None`` when no face is detected."""
        return self.predict_gaze_features(image)[:4]
