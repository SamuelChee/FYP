import math
import matplotlib.pyplot as plt
import re
import cv2
import nav
import numpy as np
import pandas as pd
import os

class OdometryEvaluation:
    def __init__(self, config):
        self.pred_path = nav.Path()
        self.gt_path_filepath = config.get('gt_path')
        self.cart_resolution = float((config.get("cart_resolution")))
        self.gt_data = self.load_gt_data() if self.gt_path_filepath else None
        self.gt_path = nav.Path()
        self.gt_timestamp_to_coord = {row['source_radar_timestamp']: (
            row['x'], row['y'], row['yaw']) for _, row in self.gt_data.iterrows()} if self.gt_data is not None else {}
        self.gt_path_length = 0
    def get_pred_path(self):
        return self.pred_path
    
    def get_gt_path(self):
        return self.gt_path
    
    def load_gt_data(self):
        if not os.path.isfile(self.gt_path_filepath):
            raise IOError(
                f"Could not find ground truth path file: {self.gt_path_filepath}")
        return pd.read_csv(self.gt_path_filepath)

    def update_path(self, tx, ty, theta, timestamp):
        if tx is None:
            return
        
        if self.gt_data is not None:
            # Check if the ground truth line has been created already
            if timestamp in self.gt_timestamp_to_coord:
                dx, dy, dtheta = self.gt_timestamp_to_coord[timestamp]
                self.gt_path.add_relative_pose(
                    nav.SE2Pose(-dy, dx, -dtheta), timestamp)
            else:
                return
                
        dx = tx * self.cart_resolution
        dy = ty * self.cart_resolution
        dtheta = (theta * (np.pi / 180.0))
        self.pred_path.add_relative_pose(nav.SE2Pose(dx, dy, dtheta))
        
                
    def calc_distance_interval_errors(self, distances_to_check, step_size):

        self.distance_interval_errors = {distance: [] for distance in distances_to_check}

        for start_pose_index in range(0, len(self.gt_path), step_size):
            for i in range(len(distances_to_check)):
                distance_to_check = distances_to_check[i]
                end_pose_index = self.gt_path.find_end_pose_by_distance(start_pose_index, distance_to_check)
                
                if end_pose_index == -1:
                    continue
                
                start_gt_pose = self.gt_path.poses[start_pose_index]
                end_gt_pose = self.gt_path.poses[end_pose_index]
                start_pred_pose = self.pred_path.poses[start_pose_index]
                end_pred_pose = self.pred_path.poses[end_pose_index]

                # Compute the relative transformations
                relative_gt_transform = start_gt_pose.relative_to(end_gt_pose)
                relative_pred_transform = start_pred_pose.relative_to(end_pred_pose)
                
                # Compute the error transformation
                relative_error_transform = relative_pred_transform.relative_to(relative_gt_transform)
                
                # Calculate rotational and translational errors
                r_err = relative_error_transform.theta/distance_to_check
                # r_err = (relative_error_transform.theta + np.pi) % (2 * np.pi) - np.pi
                t_err = np.sqrt(relative_error_transform.x**2 + relative_error_transform.y**2)/distance_to_check
                
                # Add the errors to the sequence errors dictionary
                self.distance_interval_errors[distance_to_check].append((r_err, t_err))
                

    def calc_average_errors(self):
        # Initialize the variables to store the sums of errors for overall averages
        total_translation_error = 0.0
        total_rotation_error = 0.0
        total_count = 0

        average_errors_by_distance = {}

        # Calculate the average errors for each sequence length
        for distance, errors in self.distance_interval_errors.items():
            if not errors:  # Skip if the list is empty
                continue

            # Sum up all translation and rotation errors for the current sequence length
            sum_translation_error = sum(t_err for r_err, t_err in errors)
            sum_rotation_error = sum(r_err for r_err, t_err in errors)

            # The number of error entries for the current sequence length
            count = len(errors)

            # Calculate the average translation and rotation errors for the current sequence length
            avg_translation_error = sum_translation_error / count
            avg_rotation_error = sum_rotation_error / count

            #Convert to percentage and degrees
            avg_translation_error = avg_translation_error * 100.0
            avg_rotation_error = avg_rotation_error * (180.0/np.pi) * 100.0


            # Store the averages in the dictionary
            average_errors_by_distance[distance] = (avg_rotation_error, avg_translation_error)

            # Add to the total sums for calculating the overall averages
            total_translation_error += sum_translation_error
            total_rotation_error += sum_rotation_error
            total_count += count

        # Calculate overall average translation and rotation errors
        if total_count > 0:
            overall_avg_translation_error = total_translation_error / total_count
            overall_avg_rotation_error = total_rotation_error / total_count
        else:
            overall_avg_translation_error = 0.0
            overall_avg_rotation_error = 0.0

        overall_avg_translation_error = overall_avg_translation_error * 100.0
        overall_avg_rotation_error = abs(overall_avg_rotation_error * (180.0/np.pi) * 100.0)

        return average_errors_by_distance, overall_avg_rotation_error, overall_avg_translation_error
 
