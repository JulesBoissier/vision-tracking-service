"""The HTTP contract between the vision service and the orchestrator.

``activity-orchestration-client/src/backend/clients/vision_tracking_client.py`` depends on the
routes, parameter names and response shapes tested here. A change that breaks one of these
tests needs a matching change in the orchestrator.
"""

import cv2
import pytest

from tests.conftest import make_frame


def jpeg(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def upload(frame):
    return {"file": ("frame.jpg", jpeg(frame), "image/jpeg")}


def calibrate(api, fake_predictor):
    points = [
        (10, (0, 0), (320, 240, -0.4, -0.3)),
        (20, (1920, 0), (320, 240, 0.4, -0.3)),
        (30, (0, 1080), (320, 240, -0.4, 0.3)),
        (40, (1920, 1080), (320, 240, 0.4, 0.3)),
    ]
    for code, (x, y), features in points:
        frame = fake_predictor.script(code, features)
        r = api.post("/cal_point", data={"x": x, "y": y}, files=upload(frame))
        assert r.status_code == 200
    return points


def test_health(api):
    """``GET /health`` answers 200 once the service has started.

    Why: the orchestrator blocks startup and every tracking tick on this check.
    """
    assert api.get("/health").status_code == 200


def test_predict_returns_screen_coordinates_after_calibration(api, fake_predictor):
    """``POST /predict`` with a JPEG returns ``{"prediction": [x, y]}`` in calibration pixels.

    Why: this is the request the orchestrator makes every tick. Coordinates come back in the
    same space as the ``x``/``y`` sent to ``/cal_point``, so the service never needs to know
    about monitors.
    """
    calibrate(api, fake_predictor)

    r = api.post("/predict", files=upload(make_frame(20)))

    assert r.status_code == 200
    assert r.json()["prediction"] == pytest.approx([1920, 0], abs=1)


@pytest.mark.parametrize("calibrated", [False, True])
def test_predict_without_result_returns_nulls(api, fake_predictor, calibrated):
    """With no calibration, or no face in the frame, ``/predict`` answers 200 with ``[null, null]``.

    Why: the orchestrator treats nulls as "no point of regard" and records a gap. Returning an
    error status instead would stop its tracking loop.
    """
    if calibrated:
        calibrate(api, fake_predictor)
        frame = fake_predictor.script(90, None)  # no face
    else:
        frame = fake_predictor.script(10, (320, 240, 0.0, 0.0))  # face, no calibration

    r = api.post("/predict", files=upload(frame))

    assert r.status_code == 200
    assert r.json() == {"prediction": [None, None]}


def test_reset_profile_clears_calibration(api, fake_predictor):
    """``POST /reset_profile`` discards the active calibration.

    Why: lets the user start a new calibration without restarting the container.
    """
    calibrate(api, fake_predictor)

    assert api.post("/reset_profile").status_code == 200

    assert api.post("/predict", files=upload(make_frame(10))).json() == {
        "prediction": [None, None]
    }


def test_profile_lifecycle(api, fake_predictor):
    """Profiles can be saved, listed, loaded after a reset, and deleted through the API.

    Why: this is the startup flow the orchestrator runs: list profiles, then either load one
    or calibrate and save a new one. Field names in ``/list_profiles`` are read by the
    orchestrator's profile table.
    """
    calibrate(api, fake_predictor)
    assert api.post("/save_profile", params={"name": "desk"}).status_code == 200

    profiles = api.get("/list_profiles").json()["profiles"]
    assert [p["profile_name"] for p in profiles] == ["desk"]
    assert {"id", "profile_name", "updated_at"} <= profiles[0].keys()
    profile_id = profiles[0]["id"]

    api.post("/reset_profile")
    assert api.post("/load_profile", params={"profile_id": profile_id}).status_code == 200
    assert api.post("/predict", files=upload(make_frame(40))).json()[
        "prediction"
    ] == pytest.approx([1920, 1080], abs=1)

    assert api.post("/delete_profile", params={"profile_id": profile_id}).status_code == 200
    assert api.get("/list_profiles").json()["profiles"] == []


def test_calibration_frame_without_face_is_reported_and_skipped(api, fake_predictor):
    """``/cal_point`` with no detectable face answers ``"added": false`` and calibration still works.

    Why: the calibration GUI captures on a key press, and a blink or glance away is common.
    The point is skipped instead of corrupting the profile. The status stays 200 so existing
    clients keep working; ``added`` lets a GUI ask the user to redo that dot.
    """
    calibrate(api, fake_predictor)

    r = api.post("/cal_point", data={"x": 960, "y": 540}, files=upload(fake_predictor.script(90, None)))

    assert r.status_code == 200
    assert r.json()["added"] is False
    assert api.post("/predict", files=upload(make_frame(20))).json()["prediction"] == pytest.approx(
        [1920, 0], abs=1
    )


def test_loading_unknown_profile_is_not_found_and_harmless(api, fake_predictor):
    """``/load_profile`` with an unknown id answers 404 and the active calibration keeps working.

    Why: the orchestrator passes whatever id the user typed. A 404 tells it the id was
    wrong; previously it got 200 and the service silently lost its calibration.
    """
    calibrate(api, fake_predictor)

    r = api.post("/load_profile", params={"profile_id": 9999})

    assert r.status_code == 404
    assert api.post("/predict", files=upload(make_frame(20))).json()["prediction"] == pytest.approx(
        [1920, 0], abs=1
    )


def test_predict_accepts_a_burst_of_frames(api, fake_predictor):
    """``/predict`` with several ``file`` parts returns one prediction from all of them.

    Why: the orchestrator sends a short burst per tick (several frames, one request, one
    answer). Older clients sending a single ``file`` are unaffected.
    """
    calibrate(api, fake_predictor)
    files = [
        ("file", ("a.jpg", jpeg(make_frame(10)), "image/jpeg")),
        ("file", ("b.jpg", jpeg(make_frame(20)), "image/jpeg")),
    ]

    r = api.post("/predict", files=files)

    assert r.status_code == 200
    assert r.json()["prediction"] == pytest.approx([960, 0], abs=1)


def test_undecodable_frame_is_a_client_error(api, fake_predictor):
    """A part that isn't an image answers 400 instead of failing inside the model.

    Why: a corrupt upload is the client's problem and should say so, not look like a
    server crash.
    """
    calibrate(api, fake_predictor)

    r = api.post("/predict", files={"file": ("x.jpg", b"not a jpeg", "image/jpeg")})

    assert r.status_code == 400


def test_implicit_calibration_point_is_used_but_not_saved(api, fake_predictor):
    """``/cal_point`` with ``implicit=true`` (and a burst of frames) is accepted but never saved in a profile.

    Why: the orchestrator sends clicks as implicit points. They must influence the
    current session only: a profile saved afterwards contains only the deliberate
    calibration.
    """
    calibrate(api, fake_predictor)
    api.post("/save_profile", params={"name": "desk"})
    click = fake_predictor.script(60, (320, 240, 0.0, 0.0))
    files = [("file", ("a.jpg", jpeg(click), "image/jpeg")), ("file", ("b.jpg", jpeg(click), "image/jpeg"))]

    r = api.post("/cal_point", data={"x": 1500, "y": 900, "implicit": "true"}, files=files)
    api.post("/save_profile", params={"name": "after-click"})
    profiles = {p["profile_name"]: p["id"] for p in api.get("/list_profiles").json()["profiles"]}
    api.post("/load_profile", params={"profile_id": profiles["after-click"]})
    restored = api.post("/predict", files=upload(make_frame(60))).json()["prediction"]

    assert r.json()["added"] is True
    # The saved profile holds only the 4 explicit points, which map these angles to centre.
    assert restored == pytest.approx([960, 540], abs=1)
