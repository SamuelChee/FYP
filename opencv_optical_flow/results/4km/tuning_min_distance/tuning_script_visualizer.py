import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
sys.path.append('../../../pipeline')
import nav

class TuningVisualizer:
    def __init__(self, tuning_results_folder, hyperparameter_names):
        self.tuning_results_folder = tuning_results_folder
        self.hyperparameter_names = hyperparameter_names

    def load_data_from_pickle(self, data_pickle_filepath):
        with open(data_pickle_filepath, "rb") as handle:
            data = pickle.load(handle)
        return data
    
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

    def plot_error_summary(self, error_values, output_path, error_name, error_unit, num_points=None, percentage_points=None):
        sorted_error_values = sorted(error_values.items(), key=lambda x: x[1])

        if num_points is not None:
            sorted_error_values = sorted_error_values[:num_points]
        elif percentage_points is not None:
            num_points = int(len(sorted_error_values) * percentage_points)
            sorted_error_values = sorted_error_values[:num_points]

        hyperparameter_values = [x[0] for x in sorted_error_values]
        error_values_list = [x[1] for x in sorted_error_values]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f'{error_name} vs {", ".join(self.hyperparameter_names)}')
        
        ax.set_xlabel(', '.join(self.hyperparameter_names))
        ax.set_ylabel(f'{error_name} ({error_unit})')

        x = range(len(hyperparameter_values))
        ax.plot(x, error_values_list, marker='o', color='blue')
        ax.set_xticks(x)
        ax.set_xticklabels([', '.join(map(str, values)) for values in hyperparameter_values], rotation=45, ha='right')

        fig.savefig(output_path)
        plt.close(fig)

    def plot_error_summary_combined(self, error_values, ax, error_name, error_unit, num_points=None, percentage_points=None):
        sorted_error_values = sorted(error_values.items(), key=lambda x: x[1])

        if num_points is not None:
            sorted_error_values = sorted_error_values[:num_points]
        elif percentage_points is not None:
            num_points = int(len(sorted_error_values) * percentage_points)
            sorted_error_values = sorted_error_values[:num_points]

        hyperparameter_values = [x[0] for x in sorted_error_values]
        error_values_list = [x[1] for x in sorted_error_values]

        ax.set_title(f'{error_name} vs {", ".join(self.hyperparameter_names)}')
        
        ax.set_xlabel(', '.join(self.hyperparameter_names))
        ax.set_ylabel(f'{error_name} ({error_unit})')

        x = range(len(hyperparameter_values))
        ax.plot(x, error_values_list, marker='o', color='blue')
        ax.set_xticks(x)
        ax.set_xticklabels([', '.join(map(str, values)) for values in hyperparameter_values], rotation=45, ha='right')

    def plot_combined_error_summary(self, translational_errors, rotational_errors, rmse_errors, output_path, num_points=None, percentage_points=None):
        fig, axs = plt.subplots(3, 1, figsize=(8, 18))
        fig.suptitle(f'Error Summaries vs {", ".join(self.hyperparameter_names)}', fontsize=16)

        self.plot_error_summary_combined(translational_errors, axs[0], "Translational Error", "%", num_points, percentage_points)
        self.plot_error_summary_combined(rotational_errors, axs[1], "Rotational Error", "deg/100m", num_points, percentage_points)
        self.plot_error_summary_combined(rmse_errors, axs[2], "Root Mean Square Error", "%", num_points, percentage_points)

        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.95)

        fig.savefig(output_path)
        plt.close(fig)

    def visualize(self, num_points=None, percentage_points=None):
        translational_errors = {}
        rotational_errors = {}
        rmse_errors = {}

        for folder_name in os.listdir(self.tuning_results_folder):
            result_folder = os.path.join(self.tuning_results_folder, folder_name)
            if os.path.isdir(result_folder):
                data_pickle_filepath = os.path.join(result_folder, "data.pickle")
                error_json_filepath = os.path.join(result_folder, "error_values.json")

                data = self.load_data_from_pickle(data_pickle_filepath)

                # Plot paths
                paths_plot_filepath = os.path.join(result_folder, "paths_plot.png")
                self.plot_paths(data, paths_plot_filepath)

                # Plot ATE
                ate_plot_filepath = os.path.join(result_folder, "ate_plot.png")
                self.plot_ate(data, ate_plot_filepath)

                # Load errors from JSON
                with open(error_json_filepath, "r") as f:
                    error_data = json.load(f)
                    translational_error = error_data["overall_average_errors"]["translational_error_percent"]
                    rotational_error = error_data["overall_average_errors"]["rotational_error_deg_per_100m"]
                    rmse_error = error_data["overall_average_errors"]["root_mean_square_error_percent"]

                    # Extract hyperparameter values from folder name
                    hyperparameter_values = re.findall(r"(\d+(?:\.\d+)?)", folder_name)
                    hyperparameter_values = tuple(map(float, hyperparameter_values))
                    
                    translational_errors[hyperparameter_values] = translational_error
                    rotational_errors[hyperparameter_values] = rotational_error
                    rmse_errors[hyperparameter_values] = rmse_error

        # Plot error summaries
        translational_error_plot_filepath = os.path.join(self.tuning_results_folder, "translational_error_summary.png")
        self.plot_error_summary(translational_errors, translational_error_plot_filepath, "Translational Error", "%", num_points, percentage_points)

        rotational_error_plot_filepath = os.path.join(self.tuning_results_folder, "rotational_error_summary.png")
        self.plot_error_summary(rotational_errors, rotational_error_plot_filepath, "Rotational Error", "deg/100m", num_points, percentage_points)

        rmse_error_plot_filepath = os.path.join(self.tuning_results_folder, "root_mean_square_error_summary.png")
        self.plot_error_summary(rmse_errors, rmse_error_plot_filepath, "Root Mean Square Error", "%", num_points, percentage_points)

        # Plot combined error summary
        combined_plot_filepath = os.path.join(self.tuning_results_folder, "combined_error_summary.png")
        self.plot_combined_error_summary(translational_errors, rotational_errors, rmse_errors, combined_plot_filepath, num_points, percentage_points)

if __name__ == "__main__":
    current_directory = os.getcwd()
    hyperparameter_names = ["Min Distance"]  # Replace with your actual hyperparameter names
    visualizer = TuningVisualizer(current_directory, hyperparameter_names)
    visualizer.visualize(num_points=20)  # Specify the number of points or percentage of points to plot