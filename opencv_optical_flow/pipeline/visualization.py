import math
import matplotlib.pyplot as plt
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
        result = self.setup_figure()
        if result is not None:
            self.fig, self.axes = result
        else:
            self.fig = self.axes = None


    def setup_figure(self, max_plots_per_row=3):
        if not self.enabled_visualizations:
            return

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
        if self.axes is None or "raw_radar_img" not in self.axes:
            return


        # Update the raw scan visualization
        ax = self.axes["raw_radar_img"]
        ax.clear()
        ax.imshow(raw_radar_img, cmap='gray')
        ax.set_title("Raw Radar Image")

    def update_filtered_radar_img(self, filtered_radar_img):
        if self.axes is None or "filtered_radar_img" not in self.axes:
            return

        # Update the filtered scan visualization
        ax = self.axes["filtered_radar_img"]
        ax.clear()
        ax.imshow(filtered_radar_img, cmap='gray')
        ax.set_title("Filtered Radar Image")

    def update_feature_point_img(self, feature_point_img, features):
        if self.axes is None or "feature_point_img" not in self.axes:
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
        if self.axes is None or "flow_img" not in self.axes:
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

    def update_path_plot(self, pred_path, gt_path):
        if self.axes is None or "path_plot" not in self.axes:
            return

        ax = self.axes["path_plot"]
        ax.set_xlim(-350*5, 120*5)
        ax.set_ylim(-50*5, 350*5)
        ax.set_xlabel('X(Rightward) position (m)')
        ax.set_ylabel('Y(Forward) Position (m)')
        ax.set_title('Vehicle Paths')
        # Check if the prediction line has been created already
        if hasattr(self, 'pred_line'):
            # Update the existing line
            self.pred_line.set_data(
                pred_path.x_coords(), pred_path.y_coords())
        else:
            # Create the line for the first time
            self.pred_line, = ax.plot(pred_path.x_coords(
            ), pred_path.y_coords(), color='red', label='Predicted Path', zorder=2)
        # Draw the ground truth path if it exists
        
        if hasattr(self, 'gt_line'):
            # Update the existing line
            self.gt_line.set_data(
                gt_path.x_coords(), gt_path.y_coords())
        else:
            # Create the line for the first time
            self.gt_line, = ax.plot(gt_path.x_coords(), gt_path.y_coords(
            ), color='green', label='Ground Truth Path', zorder=1)

        ax.legend(loc='upper left')

    def update_error_plot(self, pred_path, gt_path):
        # Update the error visualization

        if self.axes is None or "error_plot" not in self.axes:
            return
        ax = self.axes["error_plot"]
        ax.set_title('Absolute Trajectory Error')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Absolute Error (M)')

        errors = pred_path.calculate_ate_vector(gt_path)
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
            
        
    def show(self):
        # Redraw the canvas and show the updated figure
        if self.fig is None:
            return
        self.fig.canvas.draw()
        plt.show(block=False)
        plt.pause(1/1000.0)

    def save_figure(self, path):
        if self.fig is None:
            return
        # Save the current figure to the given path
        self.fig.savefig(path)

    def close(self):
        if self.fig is None:
            return
        # Close the Matplotlib figure
        plt.close(self.fig)









        # self.pred_path = nav.Path()
        # self.gt_data = self.load_gt_data() if self.gt_path_filepath else None
        # self.gt_path = nav.Path()
        # self.gt_timestamp_to_coord = {row['source_radar_timestamp']: (
        #     row['x'], row['y'], row['yaw']) for _, row in self.gt_data.iterrows()} if self.gt_data is not None else {}
        # # Placeholder if load_additional_path_filepaths is not implemented
        # self.additional_paths = {}

    # def load_gt_data(self):
    #     if not os.path.isfile(self.gt_path_filepath):
    #         raise IOError(
    #             f"Could not find ground truth path file: {self.gt_path_filepath}")
    #     return pd.read_csv(self.gt_path_filepath)
