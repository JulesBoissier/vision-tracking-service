import random
import unittest

from src.backend.calibration_agents import InterpolationAgent


class TestInterpolationAgent(unittest.TestCase):
    def test_calibration_step(self):
        self.ca = InterpolationAgent()

        # Adding one point with values: x = 100, y = 200, theta = 30 and phi = 45
        self.ca.calibration_step(100, 200, 50, 55, 30, 45)

        # Checking that CalibrationMap contains the correct values
        self.assertEqual(self.ca.calibration_map.monitor_x_values, [100])
        self.assertEqual(self.ca.calibration_map.monitor_y_values, [200])
        self.assertEqual(self.ca.calibration_map.head_x_values, [50])
        self.assertEqual(self.ca.calibration_map.head_y_values, [55])
        self.assertEqual(self.ca.calibration_map.theta_values, [30])
        self.assertEqual(self.ca.calibration_map.phi_values, [45])

        # Adding second point with values: x = 200, y = 5000, theta = 120 and phi = 55
        self.ca.calibration_step(200, 5000, 99, 91, 120, 55)

        # Adding third point with values: x = 1, y = 1, theta = 1 and phi = 1
        self.ca.calibration_step(1, 1, 1, 1, 1, 1)

        # Checking that second value of each list still points to the second point
        self.assertEqual(self.ca.calibration_map.monitor_x_values[1], 200)
        self.assertEqual(self.ca.calibration_map.monitor_y_values[1], 5000)
        self.assertEqual(self.ca.calibration_map.head_x_values[1], 99)
        self.assertEqual(self.ca.calibration_map.head_y_values[1], 91)
        self.assertEqual(self.ca.calibration_map.theta_values[1], 120)
        self.assertEqual(self.ca.calibration_map.phi_values[1], 55)

    def test_interpolate_to_existing_anchor(self):
        self.ca = InterpolationAgent()

        # Setting up a target angle present in calibration values
        target_angle = 1
        target_head_position = 0
        calibration_monitor_coordinates = [100, 200, 300]
        calibration_head_coordinates = [-100, 0, 100]
        calibration_angles = [0, 1, 2]

        # Verify that output matches calibration
        interpolated_coordinate = self.ca._interpolate(
            target_head_position,
            target_angle,
            calibration_monitor_coordinates,
            calibration_head_coordinates,
            calibration_angles,
        )
        self.assertAlmostEqual(interpolated_coordinate, 200)

    def test_interpolate_to_non_existing_anchor(self):
        self.ca = InterpolationAgent()

        random_monitor_coordinate = random.randint(1, 1000)
        random_head_coordinate = random.randint(1, 1000)
        random_angle = random.randint(1, 2)

        # Setting up a target angle not present in calibration values
        target_angle = 2 * random_angle
        target_head = 2 * random_head_coordinate
        calibration_monitor_coordinates = [
            random_monitor_coordinate,
            3 * random_monitor_coordinate,
        ]
        calibration_head_coordinates = [
            random_head_coordinate,
            3 * random_head_coordinate,
        ]
        calibration_angles = [random_angle, 3 * random_angle]

        # Verify that output falls in the middle of the calibration coordinates
        interpolated_coordinate = self.ca._interpolate(
            target_head,
            target_angle,
            calibration_monitor_coordinates,
            calibration_head_coordinates,
            calibration_angles,
        )
        self.assertAlmostEqual(interpolated_coordinate, 2 * random_monitor_coordinate)
