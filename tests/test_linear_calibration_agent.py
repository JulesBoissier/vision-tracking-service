"""Behaviour of ``LinearCalibrationAgent``, the default calibration model.

It maps the two gaze angles to screen coordinates with a robust linear fit, ignores head
position, and clamps predictions to the calibrated area. The reasons for each choice come
from measurements on the attention-benchmark datasets (see the class docstring). If a test
here gets in the way of a better model, check its "Why" first: each one guards a property
that was measured to matter.
"""

import random

import pytest

from src.backend.calibration_agents import LinearCalibrationAgent, NotCalibratedError
from src.backend.calibration_map import CalibrationMap

# A "true" screen mapping with some cross-coupling, as from a slightly rotated camera.
def true_screen(theta, phi):
    return 960 + 2400 * theta + 150 * phi, 600 - 2000 * phi + 100 * theta


def grid(n=5, head=(320, 240)):
    points = []
    for i in range(n):
        for j in range(n):
            theta = -0.35 + 0.7 * i / (n - 1)
            phi = -0.25 + 0.5 * j / (n - 1)
            x, y = true_screen(theta, phi)
            points.append((x, y, head[0], head[1], theta, phi))
    return points


def calibrated(points, **kwargs):
    agent = LinearCalibrationAgent(**kwargs)
    for p in points:
        agent.calibration_step(*p)
    return agent


def test_recovers_a_linear_gaze_to_screen_mapping():
    """Between calibration dots, predictions follow the linear mapping the dots came from.

    Why: on the benchmark, screen position is close to linear in the gaze angles (r=0.92 to
    0.96 per axis). Fitting both angles to each axis also absorbs a slightly rotated camera.
    """
    agent = calibrated(grid())

    for theta, phi in [(0.0, 0.0), (0.1, -0.12), (-0.2, 0.17)]:
        assert agent.calculate_point_of_regard(320, 240, theta, phi) == pytest.approx(
            true_screen(theta, phi), abs=1
        )


def test_head_position_does_not_move_the_prediction():
    """The same gaze angles predict the same point wherever the face is in the camera image.

    Why: head position from the face box jitters between frames and shifts when posture
    changes after calibration. On the multi-monitor benchmark the user sat lower during the
    test, and models using head position degraded sharply while angle-only models didn't.
    """
    agent = calibrated(grid(head=(320, 240)))

    centred = agent.calculate_point_of_regard(320, 240, 0.05, 0.05)
    moved = agent.calculate_point_of_regard(250, 300, 0.05, 0.05)

    assert moved == pytest.approx(centred)


def test_predictions_are_clamped_to_the_calibrated_area():
    """Gaze angles beyond the calibration range predict a point on the edge of the calibrated area.

    Why: the user is looking at the screens, so an off-screen prediction is always wrong and
    can't be attributed to a window. Clamping cut the benchmark's median error by ~15% and
    fixed every off-screen prediction.
    """
    points = grid()
    agent = calibrated(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x, y = agent.calculate_point_of_regard(320, 240, 2.0, -2.0)

    assert x == pytest.approx(max(xs))
    assert y == pytest.approx(max(ys))
    assert min(xs) <= agent.calculate_point_of_regard(320, 240, -2.0, 0)[0]


def test_one_bad_calibration_dot_has_little_influence():
    """A calibration dot recorded while looking elsewhere barely moves the fit.

    Why: calibration frames are captured on a key press. A blink or a glance at the keyboard
    produces a dot whose target and gaze disagree. Huber weighting limits its influence;
    plain least squares would shift every prediction toward it.
    """
    points = grid()
    x, y, hx, hy, theta, phi = points[12]  # centre dot
    points[12] = (x + 900, y + 600, hx, hy, theta, phi)

    robust = calibrated(points)
    plain = calibrated(points, iterations=1)  # one pass = ordinary least squares

    truth = true_screen(0.0, 0.0)
    robust_err = abs(robust.calculate_point_of_regard(320, 240, 0.0, 0.0)[0] - truth[0])
    plain_err = abs(plain.calculate_point_of_regard(320, 240, 0.0, 0.0)[0] - truth[0])
    assert robust_err < 5
    assert plain_err > 3 * robust_err


def test_noisy_angles_average_out():
    """With realistic frame-to-frame jitter on every calibration dot, the fit stays close.

    Why: L2CS angles jitter by ~0.013 rad between consecutive frames. A fit over many dots
    averages that out, where the old interpolation passed it straight through.
    """
    rng = random.Random(0)
    noisy = [
        (x, y, hx, hy, t + rng.gauss(0, 0.013), p + rng.gauss(0, 0.013))
        for x, y, hx, hy, t, p in grid()
    ]
    agent = calibrated(noisy)

    x, y = agent.calculate_point_of_regard(320, 240, 0.0, 0.0)

    assert abs(x - 960) < 30 and abs(y - 600) < 30


def test_single_calibration_point_predicts_that_point():
    """With one calibration point, every prediction is that point.

    Why: the fit must be well defined however few points the user has captured so far
    (a GUI may predict mid-calibration), never divide by zero or return NaN.
    """
    agent = calibrated([(700, 300, 320, 240, 0.0, 0.0)])

    assert agent.calculate_point_of_regard(320, 240, 0.3, -0.2) == pytest.approx((700, 300))


def test_uncalibrated_agent_raises_not_calibrated():
    """Before any calibration point, predicting raises ``NotCalibratedError``.

    Why: the engine turns this into a ``[null, null]`` prediction. A dedicated exception
    keeps "not calibrated" distinct from real errors.
    """
    with pytest.raises(NotCalibratedError):
        LinearCalibrationAgent().calculate_point_of_regard(320, 240, 0.0, 0.0)


def test_refits_when_a_profile_is_loaded_or_reset():
    """Replacing or resetting the calibration map takes effect on the next prediction.

    Why: profiles are loaded by assigning ``calibration_map`` and reset with
    ``initialize_cal_map``. The fit is cached, so a stale cache would keep predicting with
    the previous user's calibration.
    """
    agent = calibrated(grid())
    agent.calculate_point_of_regard(320, 240, 0.0, 0.0)  # populate the cache

    other = CalibrationMap()
    other.add_calibration_point(100, 100, 320, 240, 0.0, 0.0)
    agent.calibration_map = other
    assert agent.calculate_point_of_regard(320, 240, 0.2, 0.2) == pytest.approx((100, 100))

    agent.initialize_cal_map()
    with pytest.raises(NotCalibratedError):
        agent.calculate_point_of_regard(320, 240, 0.0, 0.0)
