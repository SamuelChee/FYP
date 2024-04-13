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



# class TuningVisualizer:
#     def __init__(self, tuning_results_folder, hyperparameter_name):
#         self.tuning_results_folder = tuning_results_folder
#         self.hyperparameter_name = hyperparameter_name

#     def load_data_from_pickle(self, data_pickle_filepath):
#         with open(data_pickle_filepath, "rb") as handle:
#             data = pickle.load(handle)
#         return data

#     def plot_paths(self, data, output_path):
#         pred_path = data["pred_path"]
#         gt_path = data["gt_path"]

#         fig, ax = plt.subplots(figsize=(8, 6))
#         ax.set_xlim(-350, 120)
#         ax.set_ylim(-50, 350)
#         ax.set_xlabel('X(Rightward) position (m)')
#         ax.set_ylabel('Y(Forward) Position (m)')
#         ax.set_title('Vehicle Paths')

#         ax.plot(pred_path.x_coords(), pred_path.y_coords(), color='red', label='Predicted Path', zorder=2)
#         ax.plot(gt_path.x_coords(), gt_path.y_coords(), color='green', label='Ground Truth Path', zorder=1)
#         ax.legend(loc='upper left')

#         fig.savefig(output_path)
#         plt.close(fig)

#     def plot_ate(self, data, output_path):
#         ate_values = data["ate"]

#         fig, ax = plt.subplots(figsize=(8, 6))
#         ax.set_title('Absolute Trajectory Error')
#         ax.set_xlabel('Frame')
#         ax.set_ylabel('Absolute Error (M)')

#         ax.plot(range(len(ate_values)), ate_values, color='red', label='Error', zorder=1)
#         ax.legend(loc='upper left')

#         fig.savefig(output_path)
#         plt.close(fig)

#     def plot_translational_error_summary(self, translational_errors, output_path):
#         hyper_parameters, error_values = zip(*sorted(translational_errors.items()))

#         fig, ax = plt.subplots(figsize=(8, 6))
#         ax.set_title('Translational Error vs ' + self.hyperparameter_name)
#         ax.set_xlabel(self.hyperparameter_name)
#         ax.set_ylabel('Translational Error (%)')

#         ax.plot(hyper_parameters, error_values, marker='o', color='blue')

#         fig.savefig(output_path)
#         plt.close(fig)

#     def visualize(self):
#         translational_errors = {}

#         for folder_name in os.listdir(self.tuning_results_folder):
#             result_folder = os.path.join(self.tuning_results_folder, folder_name)
#             if os.path.isdir(result_folder):
#                 data_pickle_filepath = os.path.join(result_folder, "data.pickle")
#                 error_json_filepath = os.path.join(result_folder, "error_values.json")

#                 data = self.load_data_from_pickle(data_pickle_filepath)
#                 # gt_path = data["gt_path"]
#                 # print(gt_path.total_path_length())
#                 # exit()
#                 # Plot paths
#                 paths_plot_filepath = os.path.join(result_folder, "paths_plot.png")
#                 self.plot_paths(data, paths_plot_filepath)

#                 # Plot ATE
#                 ate_plot_filepath = os.path.join(result_folder, "ate_plot.png")
#                 self.plot_ate(data, ate_plot_filepath)

#                 # Load translational error from JSON
#                 with open(error_json_filepath, "r") as f:
#                     error_data = json.load(f)
#                     translational_error = error_data["overall_average_errors"]["translational_error_percent"]
#                     value = None
#                     matches = re.findall(r"\d+", folder_name)
#                     if matches:
#                         value = ".".join(matches)
#                     else:
#                         exit(1)
#                     translational_errors[value] = translational_error

#         # Plot translational error summary
#         translational_error_plot_filepath = os.path.join(self.tuning_results_folder, "translational_error_summary.png")
#         self.plot_translational_error_summary(translational_errors, translational_error_plot_filepath)

class TuningVisualizer:
    def __init__(self, tuning_results_folder, hyperparameter_name):
        self.tuning_results_folder = tuning_results_folder
        self.hyperparameter_name = hyperparameter_name

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
        hyper_parameters, error_values = zip(*sorted(error_values.items(), key=lambda x: float(x[0])))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f'{error_name} vs {self.hyperparameter_name}')
        ax.set_xlabel(self.hyperparameter_name)
        ax.set_ylabel(f'{error_name} ({error_unit})')

        ax.plot(range(len(hyper_parameters)), error_values, marker='o', color='blue')
        ax.set_xticks(range(len(hyper_parameters)))
        ax.set_xticklabels(hyper_parameters, rotation=45, ha='right')

        fig.savefig(output_path)
        plt.close(fig)

    def plot_error_summary_combined(self, error_values, ax, error_name, error_unit):
        hyper_parameters, error_values = zip(*sorted(error_values.items(), key=lambda x: float(x[0])))

        ax.set_title(f'{error_name} vs {self.hyperparameter_name}')
        ax.set_xlabel(self.hyperparameter_name)
        ax.set_ylabel(f'{error_name} ({error_unit})')

        ax.plot(range(len(hyper_parameters)), error_values, marker='o', color='blue')
        ax.set_xticks(range(len(hyper_parameters)))
        ax.set_xticklabels(hyper_parameters, rotation=45, ha='right')


        # if all(isinstance(x, int) for x in hyper_parameters):
        #     ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True))
        # else:
        #     ax.xaxis.set_major_locator(MaxNLocator(nbins='auto'))

    def plot_combined_error_summary(self, translational_errors, rotational_errors, rmse_errors, output_path):
        fig, axs = plt.subplots(3, 1, figsize=(8, 18))
        fig.suptitle(f'Error Summaries vs {self.hyperparameter_name}', fontsize=16)

        self.plot_error_summary_combined(translational_errors, axs[0], "Translational Error", "%")
        self.plot_error_summary_combined(rotational_errors, axs[1], "Rotational Error", "deg/100m")
        self.plot_error_summary_combined(rmse_errors, axs[2], "Root Mean Square Error", "%")

        plt.tight_layout(pad=3.0)  # Adjust the spacing between subplots
        plt.subplots_adjust(top=0.95)  # Adjust the top spacing of the figure

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

                    value = None
                    matches = re.findall(r"\d+", folder_name)
                    if matches:
                        value = ".".join(matches)
                    else:
                        exit(1)
                    translational_errors[value] = translational_error
                    rotational_errors[value] = rotational_error
                    rmse_errors[value] = rmse_error

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
    
    pred_path, gt_path, rmse_percentage, ate_values, average_errors_by_distance, overall_avg_translation_error, overall_avg_rotation_error = pipeline.run()
    
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

# def tune_hyperparameters_window_size():
#     base_config_file = "config/pipeline_config.ini"
#     base_output_folder = "tuning_results_multithread"
    
#     window_sizes = range(5, 40)  # Window sizes from 15 to 25
    
#     threads = []
    
#     for window_size in window_sizes:
#         # Create a new config file for each window size
#         config = configparser.ConfigParser()
#         config.read(base_config_file)
#         config.set("flow_estimator", "window_size", str(window_size))
        
#         # Create a new output folder for each window size
#         output_folder = os.path.join(base_output_folder, f"window_size_{window_size}")
#         os.makedirs(output_folder, exist_ok=True)
        
#         # Save the modified config file
#         config_file = os.path.join(output_folder, "pipeline_config.ini")
#         with open(config_file, "w") as f:
#             config.write(f)
        
#         # Create a new daemon thread for each pipeline run
#         thread = threading.Thread(target=run_pipeline, args=(config_file, output_folder), daemon=True)
#         threads.append(thread)
#         thread.start()
    
#     # Wait for Ctrl+C to gracefully exit
#     signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))
#     import time
#     while True:
#         time.sleep(1)

# def tune_hyperparameters_k():
#     base_config_file = "config/pipeline_config.ini"
#     base_output_folder = "../results/tuning_k"
    
#     ks = range(5, 40)  # Window sizes from 15 to 25
    
#     threads = []
    
#     for k in ks:
#         # Create a new config file for each window size
#         config = configparser.ConfigParser()
#         config.read(base_config_file)
#         config.set("preprocessor", "k", str(k))
        
#         # Create a new output folder for each window size
#         output_folder = os.path.join(base_output_folder, f"k_{k}")
#         os.makedirs(output_folder, exist_ok=True)
        
#         # Save the modified config file
#         config_file = os.path.join(output_folder, "pipeline_config.ini")
#         with open(config_file, "w") as f:
#             config.write(f)
        
#         # Create a new daemon thread for each pipeline run
#         thread = threading.Thread(target=run_pipeline, args=(config_file, output_folder), daemon=True)
#         threads.append(thread)
#         thread.start()
    
#     # Wait for Ctrl+C to gracefully exit
#     signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))
#     import time
#     while True:
#         time.sleep(1)

# def tune_hyperparameters_z_min(base_config_file, base_output_folder):
#     z_mins = [round(z, 3) for z in np.arange(0.25, 0.35, 0.005)]

#     threads = []

    
#     for z_min in z_mins:
#         # Create a new config file for each window size
#         config = configparser.ConfigParser()
#         config.read(base_config_file)
#         config.set("preprocessor", "z_min", str(z_min))
        
#         # Create a new output folder for each window size
#         z_min_str = f"{z_min:.3f}".replace(".", "_")  # Convert z_min to a string with underscores
#         output_folder = os.path.join(base_output_folder, f"z_min_{z_min_str}")
#         os.makedirs(output_folder, exist_ok=True)
        
#         # Save the modified config file
#         config_file = os.path.join(output_folder, "pipeline_config.ini")
#         with open(config_file, "w") as f:
#             config.write(f)
        
#         # Create a new daemon thread for each pipeline run
#         thread = threading.Thread(target=run_pipeline, args=(config_file, output_folder), daemon=True)
#         threads.append(thread)
#         thread.start()

#     # Wait for Ctrl+C to gracefully exit
#     exit_event = threading.Event()
#     def signal_handler(signum, frame):
#         exit_event.set()
#     signal.signal(signal.SIGINT, signal_handler)
#     # Wait for either all threads to finish or Ctrl+C
#     while not exit_event.is_set():
#         if all(not thread.is_alive() for thread in threads):
#             break
#         time.sleep(1)
    
def tune_hyperparameters(base_config_file, base_output_folder, hyperparameter_name, hyperparameter_values, config_section, config_option):
    threads = []

    for value in hyperparameter_values:
        # Create a new config file for each hyperparameter value
        config = configparser.ConfigParser()
        config.read(base_config_file)
        config.set(config_section, config_option, str(value))

        # Create a new output folder for each hyperparameter value
        value_str = f"{value:.7f}".replace(".", "_") if isinstance(value, float) else str(value)
        output_folder = os.path.join(base_output_folder, f"{hyperparameter_name}_{value_str}")
        os.makedirs(output_folder, exist_ok=True)

        # Save the modified config file
        config_file = os.path.join(output_folder, "pipeline_config.ini")
        with open(config_file, "w") as f:
            config.write(f)

        # Create a new daemon thread for each pipeline run
        thread = threading.Thread(target=run_pipeline, args=(config_file, output_folder), daemon=True)
        threads.append(thread)
        thread.start()

    # Wait for Ctrl+C to gracefully exit
    exit_event = threading.Event()
    def signal_handler(signum, frame):
        exit_event.set()
    signal.signal(signal.SIGINT, signal_handler)
    # Wait for either all threads to finish or Ctrl+C
    while not exit_event.is_set():
        if all(not thread.is_alive() for thread in threads):
            break
        time.sleep(1)

def tune_hyperparameters_window_size(base_config_file, base_output_folder):
    window_sizes = range(5, 40)  # Window sizes from 5 to 40
    tune_hyperparameters(base_config_file, base_output_folder, "window_size", window_sizes, "flow_estimator", "window_size")

def tune_hyperparameters_k(base_config_file, base_output_folder):
    ks = range(5, 40)  # K values from 5 to 40
    tune_hyperparameters(base_config_file, base_output_folder, "k", ks, "preprocessor", "k")

def tune_hyperparameters_z_min(base_config_file, base_output_folder):
    z_mins = [round(z, 3) for z in np.arange(0.25, 0.35, 0.005)]
    tune_hyperparameters(base_config_file, base_output_folder, "z_min", z_mins, "preprocessor", "z_min")

def tune_hyperparameters_max_features(base_config_file, base_output_folder):
    max_features = range(10, 31, 1)
    tune_hyperparameters(base_config_file, base_output_folder, "max_features", max_features, "feature_detector", "max_features")

def tune_hyperparameters_quality_level(base_config_file, base_output_folder):
    quality_levels = [round(level, 3) for level in np.arange(0.001, 0.011, 0.001)]
    tune_hyperparameters(base_config_file, base_output_folder, "quality_level", quality_levels, "feature_detector", "quality_level")

def tune_hyperparameters_min_distance(base_config_file, base_output_folder):
    min_distances = range(65, 105, 5)
    tune_hyperparameters(base_config_file, base_output_folder, "min_distance", min_distances, "feature_detector", "min_distance")

def tune_hyperparameters_min_distance_narrow(base_config_file, base_output_folder):
    min_distances = range(10, 26, 1)
    tune_hyperparameters(base_config_file, base_output_folder, "min_distance", min_distances, "feature_detector", "min_distance")






#LK Param
def tune_hyperparameters_window_size(base_config_file, base_output_folder):
    window_sizes = range(3, 32, 1)
    tune_hyperparameters(base_config_file, base_output_folder, "window_size", window_sizes, "flow_estimator", "window_size")

def tune_hyperparameters_max_level(base_config_file, base_output_folder):
    max_levels = range(0, 6, 1)
    tune_hyperparameters(base_config_file, base_output_folder, "max_level", max_levels, "flow_estimator", "max_level")

def tune_hyperparameters_eig_threshold_wide(base_config_file, base_output_folder):
    np.set_printoptions(suppress=True, precision=7)
    eig_thresholds = np.logspace(-5, -3, num=20)
    eig_thresholds = np.array([np.format_float_positional(value, trim='-', precision=7) for value in eig_thresholds], dtype=float)
    tune_hyperparameters(base_config_file, base_output_folder, "eig_threshold", eig_thresholds, "flow_estimator", "min_eig_threshold")



#Ransac Param
def tune_hyperparameters_reproj_threshold_wide(base_config_file, base_output_folder):
    reproj_thresholds = [round(t, 3) for t in np.arange(0.5, 10.5, 0.5)]
    tune_hyperparameters(base_config_file, base_output_folder, "reproj_threshold", reproj_thresholds, "odometry_estimator", "reproj_threshold")

if __name__ == "__main__":

    base_config_file = "config/pipeline_config.ini"


    # hyperparameter_name = "Max Features"
    # base_output_folder = "../results/tuning_max_features_wide"
    # base_output_folder = "../results/tuning_max_features_narrow"



    # hyperparameter_name = "K"
    # base_output_folder = "../results/tuning_k"

    hyperparameter_name = "Quality Level"
    # base_output_folder = "../results/tuning_quality_level_wide"
    # base_output_folder = "../results/tuning_quality_level_narrow"
    base_output_folder = "../results/tuning_quality_level_narrow_2"


    # hyperparameter_name = "Min Distance"
    # base_output_folder = "../results/tuning_min_distance_wide"
    # base_output_folder = "../results/tuning_min_distance_narrow"

    #LK Param
    # hyperparameter_name = "Window Size"
    # base_output_folder = "../results/tuning_window_size"

    # hyperparameter_name = "Max Level"
    # base_output_folder = "../results/tuning_max_level"

    # hyperparameter_name = "Eig Threshold"
    # base_output_folder = "../results/tuning_eig_threshold_wide"

    #Ransac param:
    # hyperparameter_name = "Reproj Threshold"
    # base_output_folder = "../results/tuning_reproj_threshold"


    # hyperparameter_name = "Max Iters"
    # base_output_folder = "../results/tuning_max_iters"

    

    tuning_visualizer = TuningVisualizer(tuning_results_folder=base_output_folder, hyperparameter_name=hyperparameter_name)

    try:
        # tune_hyperparameters_max_features(base_config_file, base_output_folder)
        # tune_hyperparameters_z_min(base_config_file, base_output_folder)
        # tune_hyperparameters_k()
        # tune_hyperparameters_min_distance_narrow(base_config_file, base_output_folder)
        tune_hyperparameters_quality_level(base_config_file, base_output_folder)



        #LK Param
        # tune_hyperparameters_window_size(base_config_file=base_config_file, base_output_folder=base_output_folder)
        # tune_hyperparameters_max_level(base_config_file=base_config_file, base_output_folder=base_output_folder)
        # tune_hyperparameters_eig_threshold_wide(base_config_file=base_config_file, base_output_folder=base_output_folder)

        #Ransac Param
        # tune_hyperparameters_reproj_threshold_wide(base_config_file=base_config_file, base_output_folder=base_output_folder)
        tuning_visualizer.visualize()

    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
        sys.exit(1)  # Or perform other cleanup actions here if necessary