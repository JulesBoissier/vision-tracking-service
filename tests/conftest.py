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
def api(monkeypatch, tmp_path, fake_predictor):
    """A ``TestClient`` running the real app with a fake predictor and a temporary database.

    The app currently builds its engine inside ``lifespan`` from module-level names, so the
    seam is to patch those names. If the app moves to an app factory or dependency injection,
    only this fixture needs to change.
    """
    import src.app.routes as routes

    db_url = f"sqlite:///{tmp_path / 'api.db'}"
    monkeypatch.setattr(routes, "GazePredictor", lambda filepath: fake_predictor)
    monkeypatch.setattr(
        routes, "CalibrationProfileStore", lambda db_url_=None, **_: CalibrationProfileStore(db_url)
    )
    with TestClient(routes.app) as client:
        yield client
