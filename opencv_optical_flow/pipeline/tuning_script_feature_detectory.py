import configparser
import os
import shutil
from pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pickle
import threading
import signal
import sys
import json
import numpy as np
from tqdm import tqdm
import re
import time
from matplotlib.ticker import MaxNLocator


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

    def plot_error_summary(self, error_values, output_path, error_name, error_unit):
        hyperparameter_values = list(error_values.keys())
        error_values_list = list(error_values.values())

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

    def plot_error_summary_combined(self, error_values, ax, error_name, error_unit):
        hyperparameter_values = list(error_values.keys())
        error_values_list = list(error_values.values())

        ax.set_title(f'{error_name} vs {", ".join(self.hyperparameter_names)}')
        
        ax.set_xlabel(', '.join(self.hyperparameter_names))
        ax.set_ylabel(f'{error_name} ({error_unit})')

        x = range(len(hyperparameter_values))
        ax.plot(x, error_values_list, marker='o', color='blue')
        ax.set_xticks(x)
        ax.set_xticklabels([', '.join(map(str, values)) for values in hyperparameter_values], rotation=45, ha='right')

    def plot_combined_error_summary(self, translational_errors, rotational_errors, rmse_errors, output_path):
        fig, axs = plt.subplots(3, 1, figsize=(8, 18))
        fig.suptitle(f'Error Summaries vs {", ".join(self.hyperparameter_names)}', fontsize=16)

        self.plot_error_summary_combined(translational_errors, axs[0], "Translational Error", "%")
        self.plot_error_summary_combined(rotational_errors, axs[1], "Rotational Error", "deg/100m")
        self.plot_error_summary_combined(rmse_errors, axs[2], "Root Mean Square Error", "%")

        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.95)

        fig.savefig(output_path)
        plt.close(fig)

    def visualize(self):
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
        self.plot_error_summary(translational_errors, translational_error_plot_filepath, "Translational Error", "%")

        rotational_error_plot_filepath = os.path.join(self.tuning_results_folder, "rotational_error_summary.png")
        self.plot_error_summary(rotational_errors, rotational_error_plot_filepath, "Rotational Error", "deg/100m")

        rmse_error_plot_filepath = os.path.join(self.tuning_results_folder, "root_mean_square_error_summary.png")
        self.plot_error_summary(rmse_errors, rmse_error_plot_filepath, "Root Mean Square Error", "%")

        # Plot combined error summary
        combined_plot_filepath = os.path.join(self.tuning_results_folder, "combined_error_summary.png")
        self.plot_combined_error_summary(translational_errors, rotational_errors, rmse_errors, combined_plot_filepath)

    


def run_pipeline(config_file, output_folder):
    config = configparser.ConfigParser()
    config.read(config_file)
    pipeline = Pipeline(config=config)
    
    pred_path, gt_path, rmse_percentage, ate_values, average_errors_by_distance, overall_avg_translation_error, overall_avg_rotation_error, poses_per_second = pipeline.run()
    
    # Save the error values in a text file
    error_data = {
            "average_errors_by_distance": {
                str(distance): {
                    "translational_error_percent": round(avg_t_err, 3),
                    "rotational_error_deg_per_100m": round(avg_r_err, 3)
                }
                for distance, (avg_r_err, avg_t_err) in average_errors_by_distance.items()
            },
            "overall_average_errors": {
                "translational_error_percent": round(overall_avg_translation_error, 3),
                "rotational_error_deg_per_100m": round(overall_avg_rotation_error, 3),
                "root_mean_square_error_percent": round(rmse_percentage, 3)
            },
            "Runtime": {
                "Poses per second": round(poses_per_second, 3)
            }
        }

    with open(os.path.join(output_folder, "error_values.json"), "w") as f:
        json.dump(error_data, f, indent=4)
    
    # Save the tx, ty, theta values in a pickle file
    data = {"pred_path": pred_path, "gt_path": gt_path, "ate": ate_values}
    with open(os.path.join(output_folder, "data.pickle"), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save the last frame Matplotlib figure
    # pipeline.visualizer.save_figure(os.path.join(output_folder, "paths_plot.png"))
    pipeline.visualizer.close()

import itertools
import concurrent.futures


def tune_hyperparameters(base_config_file, base_output_folder, hyperparameters, max_threads):
    # Generate all combinations of hyperparameter values
    hyperparameter_values = [params["values"] for params in hyperparameters]
    combinations = list(itertools.product(*hyperparameter_values))

    # Create a thread pool with a maximum number of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit pipeline runs as tasks to the thread pool
        futures = []
        for combination in combinations:
            # Create a new config file for each combination of hyperparameter values
            config = configparser.ConfigParser()
            config.read(base_config_file)

            # Update config with the current combination of hyperparameter values
            for i, params in enumerate(hyperparameters):
                config.set(params["section"], params["option"], str(combination[i]))

            # Create a new output folder for each combination of hyperparameter values
            folder_name = "_".join(f"{value}" for params, value in zip(hyperparameters, combination))
            output_folder = os.path.join(base_output_folder, folder_name)
            os.makedirs(output_folder, exist_ok=True)

            # Save the modified config file
            config_file = os.path.join(output_folder, "pipeline_config.ini")
            with open(config_file, "w") as f:
                config.write(f)

            # Submit the pipeline run as a task to the thread pool
            future = executor.submit(run_pipeline, config_file, output_folder)
            futures.append(future)

        # Initialize the progress bar
        progress_bar = tqdm(total=len(futures), unit='combination', desc='Progress', position=0, leave=True)

        # Update the progress bar as futures complete
        for future in concurrent.futures.as_completed(futures):
            progress_bar.update(1)
            tqdm.write('')  # Force immediate display of progress bar update

        # Close the progress bar
        progress_bar.close()

def tune_multiple_hyperparameters(base_config_file, base_output_folder, max_threads):
    # hyperparameters = [
    #     {
    #         "name": "k",
    #         "values": range(21, 33, 1),
    #         "section": "preprocessor",
    #         "option": "k"
    #     },
    #     {
    #         "name": "z_min",
    #         "values": [round(z, 3) for z in np.arange(0.2, 0.355, 0.005)],
    #         "section": "preprocessor",
    #         "option": "z_min"
    #     }
        
    # ]

    # hyperparameters = [
    #     {
    #         "name": "max_features",
    #         "values": range(10, 30, 5),
    #         "section": "feature_detector",
    #         "option": "max_features"
    #     },
    #     {
    #         "name": "min_distance",
    #         "values": range(5, 60, 5),
    #         "section": "feature_detector",
    #         "option": "min_distance"
    #     }
    # ]

    hyperparameters = [
        {
            "name": "max_features",
            "values": range(10, 32, 1),
            "section": "feature_detector",
            "option": "max_features"
        },
        {
            "name": "quality_level",
            "values": [round(z, 3) for z in np.arange(0.006, 0.011, 0.001)],
            "section": "feature_detector",
            "option": "quality_level"
        },
        {
            "name": "min_distance",
            "values": range(15, 25, 1),
            "section": "feature_detector",
            "option": "min_distance"
        }
    ]
    # hyperparameters = [
    #         {
    #             "name": "max_features",
    #             "values": range(10, 32, 10),
    #             "section": "feature_detector",
    #             "option": "max_features"
    #         },
    #         {
    #             "name": "quality_level",
    #             "values": [round(z, 3) for z in np.arange(0.006, 0.011, 0.005)],
    #             "section": "feature_detector",
    #             "option": "quality_level"
    #         },
    #         {
    #             "name": "min_distance",
    #             "values": range(5, 62, 10),
    #             "section": "feature_detector",
    #             "option": "min_distance"
    #         }
    #     ]

    # hyperparameter_values = [params["values"] for params in hyperparameters]
    # combinations = list(itertools.product(*hyperparameter_values))
    # print(len(combinations))
    # exit()

    tune_hyperparameters(base_config_file, base_output_folder, hyperparameters, max_threads)

    # hyperparameter_names = [params["name"] for params in hyperparameters]
    # visualizer = TuningVisualizer(base_output_folder, hyperparameter_names)
    # visualizer.visualize()


if __name__ == "__main__":
    base_config_file = "config/pipeline_config.ini"
    base_output_folder = "../results/tuning_feature_detector"
    max_threads = 32

    tune_multiple_hyperparameters(base_config_file, base_output_folder, max_threads)


    
