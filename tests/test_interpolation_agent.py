"""Behaviour of ``InterpolationAgent``, the calibration model that maps gaze features to pixels.

The agent stores calibration points of the form (screen x, screen y, head x, head y, theta,
phi) and predicts a screen position for new features with inverse-distance weighting (IDW).
These tests pin the properties of that design that the rest of the system relies on. If the
calibration model is replaced (for example by a polynomial regression), revisit each test's
"Why" section to decide whether the property should be kept.
"""

import pytest

from src.backend.calibration_agents import InterpolationAgent


def calibrated_agent(points, weight=1.0) -> InterpolationAgent:
    agent = InterpolationAgent(position_interpolation_weight=weight)
    for p in points:
        agent.calibration_step(*p)
    return agent


# (monitor_x, monitor_y, head_x, head_y, theta, phi)
GRID = [
    (0, 0, 100, 100, -0.4, -0.3),
    (960, 0, 100, 100, 0.0, -0.3),
    (1920, 0, 100, 100, 0.4, -0.3),
    (0, 540, 100, 100, -0.4, 0.0),
    (960, 540, 100, 100, 0.0, 0.0),
    (1920, 540, 100, 100, 0.4, 0.0),
    (0, 1080, 100, 100, -0.4, 0.3),
    (960, 1080, 100, 100, 0.0, 0.3),
    (1920, 1080, 100, 100, 0.4, 0.3),
]


@pytest.mark.parametrize("point", GRID)
def test_reproduces_calibration_targets(point):
    """Features recorded during calibration map back to the target they were recorded at.

    Why: calibration asks the user to look at known dots. If looking at a dot again doesn't
    land on that dot, the user's trust in the profile is lost immediately. IDW gives this
    property for free: a zero distance makes that point's weight dominate.
    """
    agent = calibrated_agent(GRID)
    monitor_x, monitor_y, head_x, head_y, theta, phi = point

    x, y = agent.calculate_point_of_regard(head_x, head_y, theta, phi)

    assert x == pytest.approx(monitor_x, abs=1)
    assert y == pytest.approx(monitor_y, abs=1)


def test_interpolates_between_calibration_targets():
    """Features halfway between two calibration points predict a point halfway between them.

    Why: users look at places between the calibration dots far more often than at the dots
    themselves. The mapping has to be continuous and roughly linear between anchors.
    """
    agent = calibrated_agent(
        [(0, 0, 100, 100, -0.4, -0.3), (1000, 600, 100, 100, 0.4, 0.3)]
    )

    x, y = agent.calculate_point_of_regard(100, 100, 0.0, 0.0)

    assert x == pytest.approx(500, abs=1)
    assert y == pytest.approx(300, abs=1)


@pytest.mark.parametrize(
    "theta,phi", [(-2.0, -2.0), (2.0, 2.0), (-2.0, 2.0), (5.0, 0.0), (0.0, -5.0)]
)
def test_predictions_stay_within_calibrated_area(theta, phi):
    """Extreme gaze angles never produce a point outside the area spanned by calibration targets.

    Why: IDW returns a weighted average of calibration targets, so it can't extrapolate. That
    keeps predictions on screen (no point is attributed to a window off the monitor), at the
    cost of compressing gaze near the edges. If the model is changed to one that extrapolates,
    this test documents the guarantee that would be lost and should be replaced by explicit
    clamping.
    """
    agent = calibrated_agent(GRID)

    x, y = agent.calculate_point_of_regard(100, 100, theta, phi)

    assert 0 <= x <= 1920
    assert 0 <= y <= 1080


def test_horizontal_and_vertical_axes_are_independent():
    """Changing the vertical gaze angle (phi) doesn't move the predicted x, and vice versa.

    Why: the agent interpolates each screen axis separately (x from theta and head x, y from
    phi and head y). This keeps the model small enough to calibrate with a 3x3 grid. A joint
    2D model would need this test to be removed deliberately.
    """
    agent = calibrated_agent(GRID)

    x_low, y_a = agent.calculate_point_of_regard(100, 100, 0.2, -0.3)
    x_high, y_b = agent.calculate_point_of_regard(100, 100, 0.2, 0.3)
    _, y_left = agent.calculate_point_of_regard(100, 100, -0.4, 0.1)
    _, y_right = agent.calculate_point_of_regard(100, 100, 0.4, 0.1)

    assert x_low == pytest.approx(x_high)
    assert y_left == pytest.approx(y_right)
    assert y_a != pytest.approx(y_b)


def test_head_position_shifts_prediction_toward_matching_calibration():
    """With equal gaze angles, the calibration point recorded at the closest head position wins.

    Why: moving the head changes the gaze angles needed to look at the same target. Storing
    head position with each calibration point lets the agent prefer points recorded from a
    similar head position. ``position_interpolation_weight`` sets how strongly.
    """
    points = [
        (0, 0, 50, 100, 0.0, 0.0),  # recorded with head on the left
        (1000, 0, 150, 100, 0.0, 0.0),  # same angles, head on the right
    ]
    agent = calibrated_agent(points, weight=1.0)

    x_left, _ = agent.calculate_point_of_regard(55, 100, 0.0, 0.0)
    x_right, _ = agent.calculate_point_of_regard(145, 100, 0.0, 0.0)

    assert x_left < 500 < x_right


def test_zero_head_weight_ignores_head_position():
    """With ``position_interpolation_weight=0`` predictions depend on gaze angles only.

    Why: the weight is the single knob that switches the model between the original
    "static head" design and the head-aware design. Zero must reproduce the static version
    exactly so the two can be compared on the benchmark.
    """
    agent = calibrated_agent(GRID, weight=0.0)

    a = agent.calculate_point_of_regard(10, 10, 0.1, 0.1)
    b = agent.calculate_point_of_regard(500, 900, 0.1, 0.1)

    assert a == pytest.approx(b)


def test_reset_discards_all_calibration_points():
    """``initialize_cal_map`` starts a fresh, empty calibration.

    Why: ``/reset_profile`` uses it to let the user recalibrate without restarting the
    service. Old points must not leak into the new calibration.
    """
    agent = calibrated_agent(GRID)

    agent.initialize_cal_map()
    agent.calibration_step(700, 300, 100, 100, 0.0, 0.0)

    # With a single calibration point every prediction is that point. Any leftover point
    # from the previous calibration would pull predictions away from it.
    for theta, phi in [(-0.4, -0.3), (0.4, 0.3), (0.0, 0.0)]:
        assert agent.calculate_point_of_regard(100, 100, theta, phi) == pytest.approx(
            (700, 300)
        )
