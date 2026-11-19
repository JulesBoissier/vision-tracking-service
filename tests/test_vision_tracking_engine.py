"""Behaviour of ``VisionTrackingEngine``: calibration, prediction and profiles wired together.

The engine is the unit the HTTP routes delegate to. These tests use ``FakeGazePredictor``
(see ``conftest.py``) so they exercise the real calibration agent and profile store without
loading the neural network.
"""

import pytest

from src.backend.vision_tracking_engine import VisionTrackingEngine

CORNERS = {
    10: ((0, 0), (320, 240, -0.4, -0.3)),
    20: ((1920, 0), (320, 240, 0.4, -0.3)),
    30: ((0, 1080), (320, 240, -0.4, 0.3)),
    40: ((1920, 1080), (320, 240, 0.4, 0.3)),
}


def calibrate(engine, fake_predictor, points=CORNERS):
    for code, (screen, features) in points.items():
        frame = fake_predictor.script(code, features)
        engine.run_single_calibration_step(screen[0], screen[1], frame)


def test_predicts_calibrated_screen_position(engine, fake_predictor):
    """A frame identical to a calibration frame predicts that calibration point's screen position.

    Why: this is the end-to-end promise of calibration. It also guards the argument order
    between engine, predictor and agent, which currently relies on two renamings cancelling
    out (see CLAUDE.md). Runs with both calibration agents.
    """
    calibrate(engine, fake_predictor)

    for code, (screen, _) in CORNERS.items():
        x, y = engine.predict_gaze_position(fake_predictor.script(code, CORNERS[code][1]))
        assert (x, y) == pytest.approx(screen, abs=1)


def test_no_face_predicts_nothing(engine, fake_predictor):
    """A frame with no detectable face yields ``(None, None)`` instead of raising.

    Why: the orchestrator samples continuously and the user is often away from the desk or
    looking down. "No point of regard" is a normal result that the orchestrator records as
    a gap; it must not surface as a server error.
    """
    calibrate(engine, fake_predictor)

    assert engine.predict_gaze_position(fake_predictor.script(90, None)) == (None, None)


def test_uncalibrated_engine_predicts_nothing(engine, fake_predictor):
    """Before any calibration, predictions are ``(None, None)``.

    Why: the service starts with an empty profile and the orchestrator may call ``/predict``
    before calibration or profile loading has finished.
    """
    frame = fake_predictor.script(10, (320, 240, 0.0, 0.0))

    assert engine.predict_gaze_position(frame) == (None, None)


def test_calibration_frame_without_face_does_not_poison_profile(engine, fake_predictor):
    """A calibration frame where no face is found is skipped, and the remaining points still work.

    Why: calibration frames are captured on a key press. Blinking or looking away for one of
    them is common and shouldn't silently make the whole profile unusable.
    """
    calibrate(engine, fake_predictor)
    added = engine.run_single_calibration_step(960, 540, fake_predictor.script(90, None))

    assert added is False

    x, y = engine.predict_gaze_position(fake_predictor.script(10, CORNERS[10][1]))

    assert (x, y) == pytest.approx((0, 0), abs=1)


def test_saved_profile_restores_calibration_after_reset(engine, fake_predictor):
    """A saved profile, loaded into a fresh engine, predicts the same as the original calibration.

    Why: profiles exist so users calibrate once and reuse it across restarts of the service
    (the database lives on a Docker volume). Loading must restore the full calibration.
    """
    calibrate(engine, fake_predictor)
    probe = fake_predictor.script(50, (320, 240, 0.1, -0.1))
    expected = engine.predict_gaze_position(probe)

    engine.save_profile("desk")
    profile_id = engine.list_profiles()[0]["id"]
    fresh = VisionTrackingEngine(fake_predictor, type(engine.cal_agent)(), engine.cps)
    fresh.load_profile(profile_id)

    assert fresh.predict_gaze_position(probe) == pytest.approx(expected)


def test_profiles_can_be_listed_and_deleted(engine, fake_predictor):
    """Saved profiles appear in ``list_profiles`` with id, name and timestamp, and can be deleted.

    Why: the orchestrator shows this list to let the user choose a profile at startup.
    """
    calibrate(engine, fake_predictor)
    engine.save_profile("desk")
    engine.save_profile("sofa")

    profiles = engine.list_profiles()
    assert sorted(p["profile_name"] for p in profiles) == ["desk", "sofa"]
    assert all({"id", "profile_name", "updated_at"} <= p.keys() for p in profiles)

    engine.delete_profile(profiles[0]["id"])

    assert [p["profile_name"] for p in engine.list_profiles()] == [
        profiles[1]["profile_name"]
    ]
