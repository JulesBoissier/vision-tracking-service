import unittest
from unittest.mock import MagicMock, patch

from src.backend.gaze_predictor import GazePredictor


class TestGazePredictor(unittest.TestCase):
    @patch("src.backend.gaze_predictor.Pipeline")
    def setUp(self, mock_pipeline):
        """
        Set up a shared GazePredictor instance with a mocked Pipeline for all tests.
        """
        # Mock the return value of Pipeline
        mock_pipeline.return_value = MagicMock()

        # Create the GazePredictor instance
        self.gaze_predictor = GazePredictor(filepath="mock_weights.pth")

        # Save the mock for later assertions if needed
        self.mock_pipeline = mock_pipeline

    def test_find_bounding_box_center(self):
        bounding_box = [0, 0, 100, 100]
        image_width = 100
        # TODO: Find transformation between bboxes and actual center

    # def test_predict_gaze_vector(self, MockPipeline):
    #     """
    #     Test the predict_gaze_vector method with a mocked Pipeline.step.
    #     """
    #     # Mock Pipeline and its step method
    #     mock_pipeline_instance = MockPipeline.return_value
    #     mock_pipeline_instance.step.return_value = MagicMock(
    #         bboxes=[[10, 20, 30, 40]], pitch=[0.5], yaw=[1.0]
    #     )

    #     # Mock image input
    #     mock_image = MagicMock()
    #     mock_image.shape = (224, 224, 3)  # Simulate an image tensor with shape

    #     # Instantiate GazePredictor
    #     gaze_predictor = GazePredictor(filepath="mock_weights.pth")

    #     # Call predict_gaze_vector
    #     result = gaze_predictor.predict_gaze_vector(mock_image)

    #     # Assert the result matches the mocked values
    #     self.assertEqual(result, (184.0, 30.0, 0.5, 1.0))  # Expected x, y, pitch, yaw

    #     # Verify the step method was called once with the correct image
    #     mock_pipeline_instance.step.assert_called_once_with(mock_image)


if __name__ == "__main__":
    unittest.main()
