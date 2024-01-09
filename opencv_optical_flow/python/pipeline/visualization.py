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
        self.enabled_visualizations = [key for key, value in config.items() if value is True]
        self.gt_path_filepath = config.get('gt_path')
        self.additional_path_filepaths = {k: v for k, v in config.items() if k != 'gt_path' and not isinstance(v, bool)}
        self.fig, self.axes = self.setup_figure()
        self.update_methods = self.register_update_methods()
        self.pred_path = nav.Path()
        self.gt_data = self.load_gt_data() if self.gt_path_filepath else None
        self.gt_path = nav.Path()
        self.gt_timestamp_to_coord = {row['source_radar_timestamp']: (row['x'], row['y'], row['yaw']) for _, row in self.gt_data.iterrows()} if self.gt_data is not None else {}

        self.additional_paths = {}  # Placeholder if load_additional_path_filepaths is not implemented

    def load_gt_data(self):
        if not os.path.isfile(self.gt_path_filepath):
            raise IOError(f"Could not find ground truth path file: {self.gt_path_filepath}")
        return pd.read_csv(self.gt_path_filepath)

    def register_update_methods(self):
        pattern = re.compile(r'^update_(\w+)_img$')
        return {m.group(1): getattr(self, method_name) for method_name in dir(self) if (m := pattern.search(method_name)) and callable(getattr(self, method_name))}

    def setup_figure(self, max_plots_per_row=3):
        if not self.enabled_visualizations:
            raise ValueError("No visualizations are enabled in the configuration.")

        num_visualizations = len(self.enabled_visualizations)
        num_columns = min(num_visualizations, max_plots_per_row)
        num_rows = math.ceil(num_visualizations / max_plots_per_row)
        fig, axes_array = plt.subplots(num_rows, num_columns, squeeze=False)
        axes_flat = axes_array.flatten()
        axes = {vis: axes_flat[i] for i, vis in enumerate(self.enabled_visualizations)}

        for ax in axes_flat[len(self.enabled_visualizations):]:
            ax.set_visible(False)

        return fig, axes

    def update(self, **kwargs):
        # Update all enabled visualizations
        for vis in self.enabled_visualizations:
            if vis in kwargs:
                update_method = self.update_methods.get(vis.rstrip("_img"))
                if update_method:
                    # print(f'Calling update method for: {vis}')
                    if vis == 'feature_point_img' and 'features' in kwargs:
                        update_method(kwargs[vis], kwargs['features'], key=vis)
                    elif vis == 'flow_img' and 'old_points' in kwargs and 'new_points' in kwargs:
                        update_method(kwargs[vis], kwargs['old_points'], kwargs['new_points'], key=vis)
                    elif vis == 'path_plot' and 'tx' in kwargs and 'ty' in kwargs and 'theta' in kwargs:
                        # print("vis = path plot")
                        update_method(kwargs[vis], kwargs['tx'], kwargs['ty'],  kwargs['theta'], kwargs['timestamp'], key=vis)
                    else:
                        update_method(kwargs[vis], key=vis)
                else:
                    print(f'No update method found for: {vis}')
        
    # Update the update_*_img methods to use 'key' instead of 'idx'
    def update_raw_radar_img(self, raw_radar_img, key):
        # Update the raw scan visualization
        ax = self.axes[key]
        ax.clear()
        ax.imshow(raw_radar_img, cmap='gray')
        ax.set_title("Raw Radar Image")

    def update_filtered_radar_img(self, filtered_radar_img, key):
        # Update the filtered scan visualization
        ax = self.axes[key]
        ax.clear()
        ax.imshow(filtered_radar_img, cmap='gray')
        ax.set_title("Filtered Radar Image")

    def update_feature_point_img(self, feature_point_img, features, key):
        # Update the feature points visualization
        ax = self.axes[key]
        ax.clear()
        ax.imshow(feature_point_img, cmap='gray')
        for i in features:
            x, y = map(int, i.ravel())
            ax.scatter(x, y, s=8, color='red', marker='o')
        ax.set_title("Feature Points")

    def update_flow_img(self, img, old_points, new_points, key):
        # Update the flow visualization
        ax = self.axes[key]
        old_points = old_points.astype(int)
        new_points = new_points.astype(int)

        # Create a colored version of the image for visualization
        flow_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Draw lines and circles for each pair of old and new points
        for (x1, y1), (x2, y2) in zip(old_points, new_points):
            cv2.arrowedLine(flow_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        ax.clear()
        ax.imshow(flow_img)
        ax.set_title("Optical Flow")

    def update_path_plot_img(self, path_plot, tx, ty, theta, timestamp, key):
        # print("call update_path_plot_img")
        if tx is None:
            return
        ax = self.axes[key]
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
            self.pred_line.set_data(self.pred_path.x_coords(), self.pred_path.y_coords())
        else:
            # Create the line for the first time
            self.pred_line, = ax.plot(self.pred_path.x_coords(), self.pred_path.y_coords(), color='red', label='Predicted Path', zorder=2)
        
        line_size = len(self.pred_line.get_xdata())
        # Draw the ground truth path if it exists
        if self.gt_data is not None:
            # Check if the ground truth line has been created already

            if timestamp in self.gt_timestamp_to_coord:
                dx, dy, dtheta = self.gt_timestamp_to_coord[timestamp]
                self.gt_path.add_relative_pose(nav.SE2Pose(-dy, dx, -dtheta), timestamp)
            if hasattr(self, 'gt_line'):
                # Update the existing line
                self.gt_line.set_data(self.gt_path.x_coords(), self.gt_path.y_coords())
            else:
                # Create the line for the first time
                self.gt_line, = ax.plot(self.gt_path.x_coords(), self.gt_path.y_coords(), color='green', label='Ground Truth Path', zorder=1)
        
        ax.legend(loc='upper left')

    def show(self):
        # Redraw the canvas and show the updated figure
        self.fig.canvas.draw()
        plt.show(block=False)
        plt.pause(1/100.0)

    def save_figure(self, path):
        # Save the current figure to the given path
        self.fig.savefig(path)
    
    def close(self):
        # Close the Matplotlib figure
        plt.close(self.fig)