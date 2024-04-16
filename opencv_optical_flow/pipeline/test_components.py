import unittest
import numpy as np
from nav import SE2Pose 
import nav
from odometry_evaluation import OdometryEvaluation


class TestSE2Pose(unittest.TestCase):

    def setUp(self):
        # Setup some poses to use in the tests
        self.pose1 = SE2Pose(1.0, 2.0, np.pi/4)  # 45 degrees
        self.pose2 = SE2Pose(-1.0, 3.0, -np.pi/2) # -90 degrees

    def test_inverse(self):
        pose = SE2Pose(1.0, 2.0, np.pi / 4)
        inv_pose = pose.inverse()
        
        # Use numpy to compute the expected inverse
        transform_matrix = np.array([
            [np.cos(pose.theta), -np.sin(pose.theta), pose.x],
            [np.sin(pose.theta), np.cos(pose.theta), pose.y],
            [0, 0, 1]
        ])
        
        expected_inv_matrix = np.linalg.inv(transform_matrix)
        expected_inv_pose = SE2Pose(
            expected_inv_matrix[0, 2],
            expected_inv_matrix[1, 2],
            np.arctan2(expected_inv_matrix[1, 0], expected_inv_matrix[0, 0])
        )

        self.assertAlmostEqual(inv_pose.x, expected_inv_pose.x)
        self.assertAlmostEqual(inv_pose.y, expected_inv_pose.y)
        self.assertAlmostEqual(inv_pose.theta, expected_inv_pose.theta)

    def test_mul(self):
        pose1 = SE2Pose(1.0, 2.0, np.pi / 4)
        pose2 = SE2Pose(-1.0, 3.0, -np.pi / 2)
        
        result_pose = pose1 * pose2
        
        # Use numpy to compute the expected result
        transform_matrix_1 = np.array([
            [np.cos(pose1.theta), -np.sin(pose1.theta), pose1.x],
            [np.sin(pose1.theta), np.cos(pose1.theta), pose1.y],
            [0, 0, 1]
        ])
        
        transform_matrix_2 = np.array([
            [np.cos(pose2.theta), -np.sin(pose2.theta), pose2.x],
            [np.sin(pose2.theta), np.cos(pose2.theta), pose2.y],
            [0, 0, 1]
        ])
        
        expected_result_matrix = np.dot(transform_matrix_1, transform_matrix_2)
        expected_result_pose = SE2Pose(
            expected_result_matrix[0, 2],
            expected_result_matrix[1, 2],
            np.arctan2(expected_result_matrix[1, 0], expected_result_matrix[0, 0])
        )

        self.assertAlmostEqual(result_pose.x, expected_result_pose.x)
        self.assertAlmostEqual(result_pose.y, expected_result_pose.y)
        self.assertAlmostEqual(result_pose.theta, expected_result_pose.theta)


    def test_distance_to(self):
        pose1 = SE2Pose(1.0, 1.0, 0.0)
        pose2 = SE2Pose(4.0, 5.0, 0.0)
        
        distance = pose1.distance_to(pose2)
        
        # Use numpy to compute the expected distance
        pose1_np = np.array([pose1.x, pose1.y])
        pose2_np = np.array([pose2.x, pose2.y])
        expected_distance = np.linalg.norm(pose1_np - pose2_np)
        
        self.assertAlmostEqual(distance, expected_distance)

    def test_angle_diff_to(self):
        pose1 = SE2Pose(0.0, 0.0, np.pi/2)  # 90 degrees
        pose2 = SE2Pose(0.0, 0.0, np.pi)    # 180 degrees
        
        angle_diff = pose1.angle_diff_to(pose2)
        
        # Use numpy to compute the expected angle difference
        expected_angle_diff = np.pi/2  # Since np.pi - np.pi/2 = np.pi/2
        
        self.assertAlmostEqual(angle_diff, expected_angle_diff)

    def test_relative_to(self):
        pose1 = SE2Pose(1.0, 1.0, np.pi/4)  # 45 degrees
        pose2 = SE2Pose(2.0, 2.0, np.pi/2)  # 90 degrees
        
        relative_pose = pose1.relative_to(pose2)
        
        # Use numpy to compute the expected relative transformation
        pose1_matrix = np.array([
            [np.cos(pose1.theta), -np.sin(pose1.theta), pose1.x],
            [np.sin(pose1.theta), np.cos(pose1.theta), pose1.y],
            [0, 0, 1]
        ])
        
        pose2_matrix = np.array([
            [np.cos(pose2.theta), -np.sin(pose2.theta), pose2.x],
            [np.sin(pose2.theta), np.cos(pose2.theta), pose2.y],
            [0, 0, 1]
        ])
        
        pose1_inv_matrix = np.linalg.inv(pose1_matrix)
        expected_relative_matrix = np.dot(pose1_inv_matrix, pose2_matrix)
        
        expected_relative_pose = SE2Pose(
            expected_relative_matrix[0, 2],
            expected_relative_matrix[1, 2],
            np.arctan2(expected_relative_matrix[1, 0], expected_relative_matrix[0, 0])
        )
        
        self.assertAlmostEqual(relative_pose.x, expected_relative_pose.x)
        self.assertAlmostEqual(relative_pose.y, expected_relative_pose.y)
        self.assertAlmostEqual(relative_pose.theta, expected_relative_pose.theta)


class TestOdometryEvaluation(unittest.TestCase):
    def setUp(self):
        config = {'cart_resolution': 0.1}
        self.odom_eval = OdometryEvaluation(config)
        
    def test_calc_distance_interval_errors(self):
        # Set up Test Case 1: Zero error scenario
        self.odom_eval.gt_path.add_relative_pose(nav.SE2Pose(1, 0, 0))
        self.odom_eval.gt_path.add_relative_pose(nav.SE2Pose(1, 0, 0))
        self.odom_eval.pred_path.add_relative_pose(nav.SE2Pose(1, 0, 0))
        self.odom_eval.pred_path.add_relative_pose(nav.SE2Pose(1, 0, 0))
        self.odom_eval.calc_distance_interval_errors([1, 2], 1)
        self.assertEqual(self.odom_eval.distance_interval_errors[1], [(0, 0)])
        self.assertEqual(self.odom_eval.distance_interval_errors[2], [])

        # Clear paths for next test case
        self.odom_eval.gt_path.poses.clear()
        self.odom_eval.pred_path.poses.clear()

        # Set up Test Case 2:
        # Ground truth path
        self.odom_eval.gt_path.add_relative_pose(nav.SE2Pose(1, 0, 0))
        self.odom_eval.gt_path.add_relative_pose(nav.SE2Pose(1, 0, 0))
        self.odom_eval.gt_path.add_relative_pose(nav.SE2Pose(1, 0, 0))  # Additional pose for interval 2
        # Introduce specific errors in predicted path
        self.odom_eval.pred_path.add_relative_pose(nav.SE2Pose(1, 0.1, 0))  # 0.1 units right
        self.odom_eval.pred_path.add_relative_pose(nav.SE2Pose(1, -0.1, 0)) # 0.1 units left
        self.odom_eval.pred_path.add_relative_pose(nav.SE2Pose(1, 0, 0))     # No lateral error for the third pose
        
        # Calculate errors for intervals 1 and 2
        self.odom_eval.calc_distance_interval_errors([1, 2], 1)
        # Assertions
        self.assertEqual(self.odom_eval.distance_interval_errors[1][0], (0, 0.1)) 
        self.assertEqual(self.odom_eval.distance_interval_errors[2][0], (0, 0.05))

    def test_calc_average_errors(self):
        # Test case 1
        self.odom_eval.distance_interval_errors = {
            1: [(0, 0), (0, 0)],
            2: [(0.1, 0.1), (0.2, 0.2)]
        }
        avg_errors, overall_rot_error, overall_trans_error = self.odom_eval.calc_average_errors()
        expected_rot_error_2 = ((0.3/2)*(180/np.pi)*100 + 180) % 360 - 180
        self.assertAlmostEqual(avg_errors[2][0], expected_rot_error_2, places=6)
        self.assertAlmostEqual(avg_errors[2][1], 15.0, places=6)
        self.assertAlmostEqual(overall_rot_error, expected_rot_error_2/2, places=6)
        self.assertAlmostEqual(overall_trans_error, 7.5, places=6)
        
    def test_calculate_rmse_percentage(self):
        # Test case 1
        self.odom_eval.gt_path.add_relative_pose(nav.SE2Pose(1, 0, 0))
        self.odom_eval.gt_path.add_relative_pose(nav.SE2Pose(1, 0, 0))
        self.odom_eval.pred_path.add_relative_pose(nav.SE2Pose(1, 0, 0))
        self.odom_eval.pred_path.add_relative_pose(nav.SE2Pose(1, 0, 0))
        rmse_percentage = self.odom_eval.calculate_rmse_percentage()
        self.assertEqual(rmse_percentage, 0)

        # Test case 2
        self.odom_eval.gt_path = nav.Path()
        self.odom_eval.pred_path = nav.Path()
        self.odom_eval.gt_path.add_relative_pose(nav.SE2Pose(1, 0, 0))
        self.odom_eval.pred_path.add_relative_pose(nav.SE2Pose(1, 1, 0))

        self.odom_eval.gt_path.add_relative_pose(nav.SE2Pose(1, 0, 0))
        self.odom_eval.pred_path.add_relative_pose(nav.SE2Pose(1, 1, 0))

        exoected_rmse_percentage = (np.sqrt((1**2 + 2**2) /2 ) / 2) *100

        rmse_percentage = self.odom_eval.calculate_rmse_percentage()
        self.assertAlmostEqual(rmse_percentage, exoected_rmse_percentage, places=6)


if __name__ == '__main__':
    unittest.main()