"""Shared fixtures for the vision-tracking-service test suite.

The suite tests behaviour at three levels, from the inside out:

* ``InterpolationAgent``: the calibration maths that turns gaze features into screen pixels.
* ``VisionTrackingEngine``: calibration, prediction and profiles wired together.
* The FastAPI routes: the HTTP contract the orchestrator depends on.

None of these tests load the real L2CS-Net model. ``FakeGazePredictor`` stands in for
``GazePredictor`` and returns scripted gaze features for synthetic frames. Accuracy of the real
model is measured by the benchmark suite (``attention-benchmark``), not by unit tests.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.backend.calibration_agents import InterpolationAgent, LinearCalibrationAgent
from src.backend.calibration_profile_store import CalibrationProfileStore
from src.backend.vision_tracking_engine import VisionTrackingEngine

Features = Tuple[float, float, float, float]  # head_x, head_y, theta, phi


def make_frame(code: int) -> np.ndarray:
    """Return a small solid-colour BGR frame whose pixel value identifies it.

    Solid colours survive JPEG round-trips almost unchanged, so the same frame can be sent
    through the HTTP API and still be recognised by ``FakeGazePredictor``.
    """
    return np.full((48, 64, 3), code, dtype=np.uint8)


class FakeGazePredictor:
    """Drop-in replacement for ``GazePredictor`` with scripted outputs.

    Frames are identified by their (rounded) mean pixel value. Codes must be multiples of 10
    so JPEG noise can't make two frames collide. A code mapped to ``None`` simulates a frame
    where no face is detected, which the real predictor reports as four ``None`` values.
    """

    def __init__(self):
        self.features: Dict[int, Optional[Features]] = {}

    def script(self, code: int, features: Optional[Features]) -> np.ndarray:
        assert code % 10 == 0 and 0 <= code <= 250
        self.features[code] = features
        return make_frame(code)

    def predict_gaze_vector(self, image: np.ndarray):
        code = int(round(float(image.mean()) / 10.0) * 10)
        features = self.features[code]
        if features is None:
            return None, None, None, None
        return features


@pytest.fixture
def fake_predictor() -> FakeGazePredictor:
    return FakeGazePredictor()


@pytest.fixture
def store(tmp_path) -> CalibrationProfileStore:
    return CalibrationProfileStore(db_url=f"sqlite:///{tmp_path / 'calibration.db'}")


@pytest.fixture(params=["idw", "linear"])
def engine(request, fake_predictor, store) -> VisionTrackingEngine:
    """An engine with each calibration agent: engine behaviour must not depend on the agent."""
    agent = InterpolationAgent() if request.param == "idw" else LinearCalibrationAgent()
    return VisionTrackingEngine(fake_predictor, agent, store)


@pytest.fixture
def api(tmp_path, fake_predictor):
    """A ``TestClient`` for an app built around an injected engine.

    The engine has the default calibration agent, a fake predictor and a temporary
    database. ``create_app(engine=...)`` replaces the monkey-patching this fixture needed
    when the app was a module-level global.
    """
    from src.app.routes import create_app

    store = CalibrationProfileStore(db_url=f"sqlite:///{tmp_path / 'api.db'}")
    engine = VisionTrackingEngine(fake_predictor, LinearCalibrationAgent(), store)
    with TestClient(create_app(engine=engine)) as client:
        yield client
