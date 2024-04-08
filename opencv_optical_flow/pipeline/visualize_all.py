import os
import pickle
import json
import matplotlib.pyplot as plt
import nav
import pandas as pd

class Visualizer:

    def load_data_from_pickle(self, data_pickle_filepath):
        with open(data_pickle_filepath, "rb") as handle:
            data = pickle.load(handle)
        return data

    # def load_gt_data(self):
    #     if not os.path.isfile(self.gt_path_filepath):
    #         raise IOError(
    #             f"Could not find ground truth path file: {self.gt_path_filepath}")
    #     return pd.read_csv(self.gt_path_filepath)

    def plot_paths(self, data, output_path):
        pred_path = data["pred_path"]
        gt_path = data["gt_path"]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-350, 120)
        ax.set_ylim(-50, 350)
        ax.set_xlabel('X(Rightward) position (m)')
        ax.set_ylabel('Y(Forward) Position (m)')
        ax.set_title('Vehicle Paths')

        ax.plot(pred_path.x_coords(), pred_path.y_coords(), color='red', label='Predicted Path', zorder=2)
        ax.plot(gt_path.x_coords(), gt_path.y_coords(), color='green', label='Ground Truth Path', zorder=1)
        ax.legend(loc='upper left')

        fig.savefig(output_path)
        plt.close(fig)

    def plot_ate(self, data, output_path):
        ate_values = data["ate"]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Absolute Trajectory Error')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Absolute Error (M)')

        ax.plot(range(len(ate_values)), ate_values, color='red', label='Error', zorder=1)
        ax.legend(loc='upper left')

        fig.savefig(output_path)
        plt.close(fig)

    def plot_translational_error_summary(self, translational_errors, output_path):
        # Sort the dictionary by keys (window sizes) and then unzip into two lists
        window_sizes, error_values = zip(*sorted(translational_errors.items()))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Translational Error vs K')
        ax.set_xlabel('k')
        ax.set_ylabel('Translational Error (%)')

        # Plot the sorted window sizes against the error values
        ax.plot(window_sizes, error_values, marker='o', color='blue')

        # Save the figure
        fig.savefig(output_path)
        plt.close(fig)

def main():
    tuning_results_folder = "tuning_results_multithread"

    visualizer = Visualizer()

    translational_errors = {}

    for folder_name in os.listdir(tuning_results_folder):
        if folder_name.startswith("k_"):
            result_folder = os.path.join(tuning_results_folder, folder_name)
            data_pickle_filepath = os.path.join(result_folder, "data.pickle")
            error_json_filepath = os.path.join(result_folder, "error_values.json")

            data = visualizer.load_data_from_pickle(data_pickle_filepath)

            # Plot paths
            paths_plot_filepath = os.path.join(result_folder, "paths_plot.png")
            visualizer.plot_paths(data, paths_plot_filepath)

            # Plot ATE
            ate_plot_filepath = os.path.join(result_folder, "ate_plot.png")
            visualizer.plot_ate(data, ate_plot_filepath)

            # Load translational error from JSON
            with open(error_json_filepath, "r") as f:
                error_data = json.load(f)
                translational_error = error_data["overall_average_errors"]["translational_error_percent"]
                window_size = int(folder_name.split("_")[-1])
                translational_errors[window_size] = translational_error

    # Plot translational error summary
    translational_error_plot_filepath = os.path.join(tuning_results_folder, "translational_error_summary.png")
    visualizer.plot_translational_error_summary(translational_errors, translational_error_plot_filepath)

if __name__ == "__main__":
    main()