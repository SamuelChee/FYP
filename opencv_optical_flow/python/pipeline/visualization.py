import math
import matplotlib.pyplot as plt
import re
import cv2
import nav
import numpy as np
import pandas as pd
import os


class Visualizer:
    def __init__(self, config):
        self.config = config
        self.enabled_visualizations = [
            key for key, value in config.items() if value is True]
        self.gt_path_filepath = config.get('gt_path')
        self.additional_path_filepaths = {
            k: v for k, v in config.items() if k != 'gt_path' and not isinstance(v, bool)}
        self.fig, self.axes = self.setup_figure()
        self.pred_path = nav.Path()
        self.gt_data = self.load_gt_data() if self.gt_path_filepath else None
        self.gt_path = nav.Path()
        self.gt_timestamp_to_coord = {row['source_radar_timestamp']: (
            row['x'], row['y'], row['yaw']) for _, row in self.gt_data.iterrows()} if self.gt_data is not None else {}
        # Placeholder if load_additional_path_filepaths is not implemented
        self.additional_paths = {}

    def load_gt_data(self):
        if not os.path.isfile(self.gt_path_filepath):
            raise IOError(
                f"Could not find ground truth path file: {self.gt_path_filepath}")
        return pd.read_csv(self.gt_path_filepath)

    def setup_figure(self, max_plots_per_row=3):
        if not self.enabled_visualizations:
            raise ValueError(
                "No visualizations are enabled in the configuration.")

        num_visualizations = len(self.enabled_visualizations)
        num_columns = min(num_visualizations, max_plots_per_row)
        num_rows = math.ceil(num_visualizations / max_plots_per_row)
        fig, axes_array = plt.subplots(num_rows, num_columns, squeeze=False)
        axes_flat = axes_array.flatten()
        axes = {vis: axes_flat[i]
                for i, vis in enumerate(self.enabled_visualizations)}

        for ax in axes_flat[len(self.enabled_visualizations):]:
            ax.set_visible(False)

        return fig, axes


    def update_raw_radar_img(self, raw_radar_img):
        if "raw_radar_img" not in self.axes:
            return


        # Update the raw scan visualization
        ax = self.axes["raw_radar_img"]
        ax.clear()
        ax.imshow(raw_radar_img, cmap='gray')
        ax.set_title("Raw Radar Image")

    def update_filtered_radar_img(self, filtered_radar_img):
        if "filtered_radar_img" not in self.axes:
            return

        # Update the filtered scan visualization
        ax = self.axes["filtered_radar_img"]
        ax.clear()
        ax.imshow(filtered_radar_img, cmap='gray')
        ax.set_title("Filtered Radar Image")

    def update_feature_point_img(self, feature_point_img, features):
        if "feature_point_img" not in self.axes:
            return
        # Update the feature points visualization
        ax = self.axes["feature_point_img"]
        ax.clear()
        ax.imshow(feature_point_img, cmap='gray')
        for i in features:
            x, y = map(int, i.ravel())
            ax.scatter(x, y, s=8, color='red', marker='o')
        ax.set_title("Feature Points")

    def update_flow_img(self, flow_img, old_points, new_points):
        if "flow_img" not in self.axes:
            return
        # Update the flow visualization
        ax = self.axes["flow_img"]
        old_points = old_points.astype(int)
        new_points = new_points.astype(int)

        # Create a colored version of the image for visualization
        flow_img = cv2.cvtColor(flow_img, cv2.COLOR_GRAY2BGR)

        # Draw lines and circles for each pair of old and new points
        for (x1, y1), (x2, y2) in zip(old_points, new_points):
            cv2.arrowedLine(flow_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        ax.clear()
        ax.imshow(flow_img)
        ax.set_title("Optical Flow")

    def update_path_plot(self, tx, ty, theta, timestamp):
        if "path_plot" not in self.axes:
            return

        if tx is None:
            return
        ax = self.axes["path_plot"]
        ax.set_xlim(-350, 120)
        ax.set_ylim(-50, 350)
        ax.set_xlabel('X(Rightward) position (m)')
        ax.set_ylabel('Y(Forward) Position (m)')
        ax.set_title('Vehicle Paths')
        cart_resolution = 0.08
        dx = tx * cart_resolution
        dy = ty * cart_resolution
        dtheta = (theta * (np.pi / 180.0))
        self.pred_path.add_relative_pose(nav.SE2Pose(dx, dy, dtheta))

        # Check if the prediction line has been created already
        if hasattr(self, 'pred_line'):
            # Update the existing line
            self.pred_line.set_data(
                self.pred_path.x_coords(), self.pred_path.y_coords())
        else:
            # Create the line for the first time
            self.pred_line, = ax.plot(self.pred_path.x_coords(
            ), self.pred_path.y_coords(), color='red', label='Predicted Path', zorder=2)

        line_size = len(self.pred_line.get_xdata())
        # Draw the ground truth path if it exists
        if self.gt_data is not None:
            # Check if the ground truth line has been created already

            if timestamp in self.gt_timestamp_to_coord:
                dx, dy, dtheta = self.gt_timestamp_to_coord[timestamp]
                self.gt_path.add_relative_pose(
                    nav.SE2Pose(-dy, dx, -dtheta), timestamp)
            if hasattr(self, 'gt_line'):
                # Update the existing line
                self.gt_line.set_data(
                    self.gt_path.x_coords(), self.gt_path.y_coords())
            else:
                # Create the line for the first time
                self.gt_line, = ax.plot(self.gt_path.x_coords(), self.gt_path.y_coords(
                ), color='green', label='Ground Truth Path', zorder=1)

        ax.legend(loc='upper left')

    def update_error_plot(self):
        # Update the error visualization
        ax = self.axes["error_plot"]
        ax.set_title('Absolute Trajectory Error')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Absolute Error (M)')

        # Calculate the absolute trajectory errors if the ground truth data is available
        if self.gt_data is not None:
            # Calculate errors using the calculate_ate_vector method
            # Assuming self.pred_path and self.gt_path are instances of the Path class
            errors = self.pred_path.calculate_ate_vector(self.gt_path)

            # Plot the errors over time
            if hasattr(self, 'error_line'):
                # Update the existing line
                self.error_line.set_data(range(len(errors)), errors)
                ax.relim()
                ax.autoscale_view()
            else:
                # Create the line for the first time
                self.error_line, = ax.plot(
                    range(len(errors)), errors, color='red', label='Error', zorder=1)

            ax.legend(loc='upper left')
        else:
            print("Ground truth data is not available for error calculation.")
            
        def calculate_errors(self, gt_path):
            assert len(self.poses) == len(gt_path.poses), "The number of poses must be the same in both paths."

            translation_errors = []
            rotation_errors = []

            for i in range(1, len(gt_path.poses)):
                # Compute the relative poses for ground truth and estimated path
                gt_rel_pose = nav.SE2Transform(
                    gt_path.poses[i].x - gt_path.poses[i-1].x,
                    gt_path.poses[i].y - gt_path.poses[i-1].y,
                    gt_path.poses[i].theta - gt_path.poses[i-1].theta
                )
                est_rel_pose = nav.SE2Transform(
                    self.poses[i].x - self.poses[i-1].x,
                    self.poses[i].y - self.poses[i-1].y,
                    self.poses[i].theta - self.poses[i-1].theta
                )

                # Ground truth distance for this segment
                distance = gt_path.poses[i-1].distance_to(gt_path.poses[i])

                # Calculate the translational error
                translation_diff = gt_rel_pose.distance_to(est_rel_pose)
                translation_error = 100 * translation_diff / distance
                translation_errors.append(translation_error)

                # Calculate the rotational error (in radians)
                rotation_diff = np.abs(gt_rel_pose.theta - est_rel_pose.theta)
                # Convert rotational difference to degrees and normalize by distance
                rotation_error_deg = np.degrees(rotation_diff) / distance
                rotation_errors.append(rotation_error_deg)

            # Compute mean errors
            mean_translation_error = np.mean(translation_errors)
            mean_rotation_error = np.mean(rotation_errors)

            return {
                'mean_translation_error_percent': mean_translation_error,
                'mean_rotation_error_deg_per_meter': mean_rotation_error
            }
        
    def show(self):
        # Redraw the canvas and show the updated figure
        self.fig.canvas.draw()
        plt.show(block=False)
        plt.pause(1/1000.0)

    def save_figure(self, path):
        # Save the current figure to the given path
        self.fig.savefig(path)

    def close(self):
        # Close the Matplotlib figure
        plt.close(self.fig)
