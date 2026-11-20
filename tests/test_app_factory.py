"""Behaviour of the app factory (``create_app``) and ``Settings``."""

import pytest
from fastapi.testclient import TestClient

import src.app.routes as routes
from src.app.routes import create_app
from src.backend.calibration_agents import InterpolationAgent, LinearCalibrationAgent
from src.backend.calibration_profile_store import CalibrationProfileStore
from src.backend.vision_tracking_engine import VisionTrackingEngine
from src.settings import Settings
from tests.conftest import make_frame


def test_settings_defaults_and_environment():
    """Settings default to the linear agent, the bundled model and a local database, and read the environment.

    Why: the Docker image points ``CALIBRATION_DB_URL`` into its data volume, and the
    benchmark switches ``CALIBRATION_AGENT`` for A/B runs. Nothing else reads the environment.
    """
    assert Settings.from_env({}) == Settings()
    assert Settings.from_env({"CALIBRATION_DISTANCE_TERM": "0"}).calibration_distance_term is False
    assert Settings().calibration_distance_term is True
    s = Settings.from_env(
        {"CALIBRATION_AGENT": "IDW", "MODEL_PATH": "/m.pkl", "CALIBRATION_DB_URL": "sqlite:////d/c.db"}
    )
    assert (s.calibration_agent, s.model_path, s.calibration_db_url) == (
        "idw",
        "/m.pkl",
        "sqlite:////d/c.db",
    )


def test_apps_are_independent(tmp_path, fake_predictor):
    """Two apps built in one process each serve their own engine and calibration.

    Why: the old module-level app kept its engine in a global dict, so there could only be
    one. Independent apps make tests isolated and allow side-by-side comparisons (e.g. the
    two calibration agents) in one process.
    """
    def app_with_point(name, x):
        store = CalibrationProfileStore(db_url=f"sqlite:///{tmp_path / name}")
        engine = VisionTrackingEngine(fake_predictor, LinearCalibrationAgent(), store)
        frame = fake_predictor.script(10, (320, 240, 0.0, 0.0))
        engine.run_single_calibration_step(x, 0, frame)
        return create_app(engine=engine)

    with TestClient(app_with_point("a.db", 100)) as a, TestClient(app_with_point("b.db", 900)) as b:
        import cv2

        jpeg = cv2.imencode(".jpg", make_frame(10))[1].tobytes()
        upload = {"file": ("f.jpg", jpeg, "image/jpeg")}
        assert a.post("/predict", files=upload).json()["prediction"][0] == pytest.approx(100)
        assert b.post("/predict", files=upload).json()["prediction"][0] == pytest.approx(900)


@pytest.mark.parametrize("name,agent_type", [("linear", LinearCalibrationAgent), ("idw", InterpolationAgent)])
def test_engine_is_built_from_settings(monkeypatch, tmp_path, fake_predictor, name, agent_type):
    """Without an injected engine, one is built at startup from the settings: agent, model path, database.

    Why: this is the production path (``uvicorn main:app``). The model is replaced here so
    the test doesn't need the 95 MB weights; that's the one remaining patch, and it only
    stands in for the network.
    """
    seen = {}

    def fake_gaze_predictor(filepath):
        seen["model_path"] = filepath
        return fake_predictor

    monkeypatch.setattr(routes, "GazePredictor", fake_gaze_predictor)
    settings = Settings(
        calibration_agent=name, model_path="/weights.pkl", calibration_db_url=f"sqlite:///{tmp_path / 's.db'}"
    )

    app = create_app(settings=settings)
    with TestClient(app):
        engine = app.state.engine
        assert isinstance(engine.cal_agent, agent_type)
        assert seen["model_path"] == "/weights.pkl"
        assert (tmp_path / "s.db").exists()
