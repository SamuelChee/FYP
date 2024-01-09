import math
import matplotlib.pyplot as plt
import re
import cv2
class Visualizer:
    def __init__(self, config):
        self.config = config
        self.enabled_visualizations = [key for key, value in self.config.items() if (isinstance(value, bool)) and value]
        self.fig, self.axes = self.setup_figure()
        self.update_methods = self._register_update_methods()
    def _register_update_methods(self):
        pattern = re.compile(r'^update_(\w+)$')
        return {
            match.group(1): getattr(self, method_name)
            for method_name in dir(self)
            if callable(getattr(self, method_name)) and (match := pattern.search(method_name))
        }
    def setup_figure(self, max_plots_per_row=3):
        # Determine the number of enabled visualizations
        num_visualizations = len(self.enabled_visualizations)
        # Calculate the number of rows and columns based on the maximum plots per row
        num_columns = min(num_visualizations, max_plots_per_row)
        num_rows = math.ceil(num_visualizations / max_plots_per_row)

        # Create a figure with a grid layout based on the number of visualizations
        if num_visualizations == 0:
            raise ValueError("No visualizations are enabled in the configuration.")
        else:
            fig, axes = plt.subplots(num_rows, num_columns, squeeze=False)
            # Flatten the axes array
            axes = axes.flatten()
            # Turn off the axes that will not be used
            for idx, ax in enumerate(axes):
                if idx >= num_visualizations:
                    ax.set_visible(False)  # This makes the extra axes invisible
            # Make sure axes is a list before returning
            axes = axes.tolist()
        return fig, axes

    def update(self, **kwargs):
        # Update all enabled visualizations
        for i, vis in enumerate(self.enabled_visualizations):
            if vis in kwargs:
                update_method = self.update_methods.get(vis)
                if update_method:
                    # print(f'Calling update method for: {vis}')
                    if vis == 'feature_point_img' and 'features' in kwargs:
                        update_method(kwargs[vis], kwargs['features'], idx=i)
                    elif vis == 'flow_img' and 'old_points' in kwargs and 'new_points' in kwargs:
                        update_method(kwargs[vis], kwargs['old_points'], kwargs['new_points'], idx=i)
                    else:
                        update_method(kwargs[vis], idx=i)
                else:
                    print(f'No update method found for: {vis}')
        
    def update_raw_radar_img(self, raw_radar_img, idx):
        # Update the raw scan visualization
        self.axes[idx].clear()
        self.axes[idx].imshow(raw_radar_img, cmap='gray')
        self.axes[idx].set_title("Raw Radar Image")

    def update_filtered_radar_img(self, filtered_radar_img, idx):
        # Update the filtered scan visualization
        
        # Make sure the axis is cleared and then redraw
        self.axes[idx].clear()
        self.axes[idx].imshow(filtered_radar_img, cmap='gray')
        self.axes[idx].set_title("Filtered Radar Image")

    def update_feature_point_img(self, feature_point_img, features, idx):
        # Update the feature points visualization
        
        # Make sure the axis is cleared and then redraw
        self.axes[idx].clear()
        self.axes[idx].imshow(feature_point_img, cmap='gray')
        for i in features:
            x, y = map(int, i.ravel())
            self.axes[idx].scatter(x, y, s=5, color='green', marker='x')
        self.axes[idx].set_title("Feature Points")

    def update_flow_img(self, img, old_points, new_points, idx):
        # Update the flow visualization
        old_points = old_points.astype(int)
        new_points = new_points.astype(int)

        # Create a colored version of the image for visualization
        flow_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Draw lines and circles for each pair of old and new points
        for (x1, y1), (x2, y2) in zip(old_points, new_points):
            cv2.arrowedLine(flow_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # cv2.circle(flow_img, (x1, y1), 2, (0, 255, 0), -1)

        # Make sure the axis is cleared and then redraw
        self.axes[idx].clear()
        self.axes[idx].imshow(flow_img)
        # for (x1, y1), (x2, y2) in zip(old_points, new_points):
        #     self.axes[idx].arrow(x1, y1, x2-x1, y2-y1,
        #                          head_width=5, head_length=10, fc='lightblue', ec='black')
        self.axes[idx].set_title("Optical Flow")

    def show(self):
        # Redraw the canvas and show the updated figure
        self.fig.canvas.draw()
        plt.show(block=False)
        plt.pause(1/60.0)

    def save_figure(self, path):
        # Save the current figure to the given path
        self.fig.savefig(path)
    
    def close(self):
        # Close the Matplotlib figure
        plt.close(self.fig)