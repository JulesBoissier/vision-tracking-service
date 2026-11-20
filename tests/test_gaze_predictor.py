"""Behaviour of ``GazePredictor``, the wrapper that turns L2CS-Net output into gaze features.

The L2CS ``Pipeline`` is replaced by a stub so these tests check only what the wrapper adds:
which face is used, how head position is derived, and how "no face" is reported. Model
accuracy is covered by the benchmark suite.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.backend.gaze_predictor import GazePredictor

IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


def predictor_returning(result=None, error=None) -> GazePredictor:
    with patch("src.backend.gaze_predictor.Pipeline") as pipeline_cls:
        pipeline = pipeline_cls.return_value
        if error:
            pipeline.step.side_effect = error
        else:
            pipeline.step.return_value = result
        return GazePredictor(filepath="unused.pkl")


def l2cs_result(bboxes, pitch, yaw):
    return SimpleNamespace(
        bboxes=np.array(bboxes, dtype=float),
        pitch=np.array(pitch, dtype=float),
        yaw=np.array(yaw, dtype=float),
    )


def test_returns_head_position_and_gaze_angles():
    """The wrapper returns (head x, head y, pitch, yaw) for the detected face.

    Why: these four numbers are the only features the calibration agent sees. Head position
    is the centre of the face bounding box.
    """
    result = l2cs_result([[100, 50, 200, 150]], pitch=[0.25], yaw=[-0.1])
    predictor = predictor_returning(result)

    head_x, head_y, pitch, yaw = predictor.predict_gaze_vector(IMAGE)

    assert head_y == pytest.approx(100)
    assert (pitch, yaw) == pytest.approx((0.25, -0.1))
    assert head_x == pytest.approx(640 - 150)


def test_head_x_is_mirrored():
    """A face on the left of the camera image has a larger head x than one on the right.

    Why: the webcam faces the user, so the image is a mirror of the user's point of view.
    Mirroring x makes "user moves right" increase head x, the same direction as screen x.
    """
    left = predictor_returning(l2cs_result([[0, 0, 100, 100]], [0], [0]))
    right = predictor_returning(l2cs_result([[540, 0, 640, 100]], [0], [0]))

    assert left.predict_gaze_vector(IMAGE)[0] > right.predict_gaze_vector(IMAGE)[0]


def test_uses_first_detected_face():
    """When several faces are detected, the first one returned by L2CS is used.

    Why: the system tracks a single user. L2CS orders detections by confidence, so the first
    face is the most likely to be the user sitting at the screen.
    """
    result = l2cs_result(
        [[100, 50, 200, 150], [400, 50, 500, 150]], pitch=[0.3, -0.3], yaw=[0.1, -0.1]
    )

    assert predictor_returning(result).predict_gaze_vector(IMAGE)[2:] == pytest.approx(
        (0.3, 0.1)
    )


def test_no_face_returns_nones():
    """When L2CS finds no face it raises ``ValueError``; the wrapper returns four ``None`` values.

    Why: "no face" is a normal outcome (user away or looking down) that callers handle as a
    missing prediction, not as an error.
    """
    predictor = predictor_returning(error=ValueError("no face"))

    assert predictor.predict_gaze_vector(IMAGE) == (None, None, None, None)


def test_reports_face_width_as_a_distance_cue():
    """``predict_gaze_features`` adds the face box width in pixels.

    Why: face width shrinks as the user leans back. It is the calibration's distance cue,
    the only one available from a single webcam frame without camera calibration.
    """
    predictor = predictor_returning(l2cs_result([[100, 50, 230, 150]], [0.1], [0.2]))

    features = predictor.predict_gaze_features(IMAGE)

    assert features[4] == pytest.approx(130)
    assert predictor.predict_gaze_vector(IMAGE) == features[:4]
