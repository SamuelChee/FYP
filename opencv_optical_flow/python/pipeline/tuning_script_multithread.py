import configparser
import os
import shutil
from pipeline import Pipeline
import numpy as np
import pickle
import threading
import signal
import sys
import json
def run_pipeline(config_file, output_folder):
    config = configparser.ConfigParser()
    config.read(config_file)
    pipeline = Pipeline(config=config)
    
    pred_path, gt_path, ate_values, average_errors_by_distance, overall_avg_translation_error, overall_avg_rotation_error = pipeline.run()
    
    # Save the error values in a text file
    error_data = {
            "average_errors_by_distance": {
                str(distance): {
                    "translation_error_percent": round(avg_t_err * 100, 3),
                    "rotation_error_deg_per_100m": round(avg_r_err / np.pi * 180 * 100, 3)
                }
                for distance, (avg_r_err, avg_t_err) in average_errors_by_distance.items()
            },
            "overall_average_errors": {
                "translational_error_percent": round(overall_avg_translation_error * 100, 3),
                "rotational_error_deg_per_100m": round(overall_avg_rotation_error * (180 / np.pi) * 100, 3)
            }
        }

    with open(os.path.join(output_folder, "error_values.json"), "w") as f:
        json.dump(error_data, f, indent=4)
    
    # Save the tx, ty, theta values in a pickle file
    data = {"pred_path": pred_path, "gt_path": gt_path, "ate": ate_values}
    with open(os.path.join(output_folder, "data.pickle"), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save the last frame Matplotlib figure
    # pipeline.visualizer.save_figure(os.path.join(output_folder, "last_frame_figure.png"))
    # pipeline.visualizer.close()

def tune_hyperparameters_window_size():
    base_config_file = "config/pipeline_config.ini"
    base_output_folder = "tuning_results_multithread"
    
    window_sizes = range(5, 40)  # Window sizes from 15 to 25
    
    threads = []
    
    for window_size in window_sizes:
        # Create a new config file for each window size
        config = configparser.ConfigParser()
        config.read(base_config_file)
        config.set("flow_estimator", "window_size", str(window_size))
        
        # Create a new output folder for each window size
        output_folder = os.path.join(base_output_folder, f"window_size_{window_size}")
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
    signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))
    import time
    while True:
        time.sleep(1)

def tune_hyperparameters_window_size():
    base_config_file = "config/pipeline_config.ini"
    base_output_folder = "tuning_results_multithread"
    
    window_sizes = range(5, 40)  # Window sizes from 15 to 25
    
    threads = []
    
    for window_size in window_sizes:
        # Create a new config file for each window size
        config = configparser.ConfigParser()
        config.read(base_config_file)
        config.set("flow_estimator", "window_size", str(window_size))
        
        # Create a new output folder for each window size
        output_folder = os.path.join(base_output_folder, f"window_size_{window_size}")
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
    signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))
    import time
    while True:
        time.sleep(1)

def tune_hyperparameters_k():
    base_config_file = "config/pipeline_config.ini"
    base_output_folder = "tuning_results_multithread"
    
    ks = range(5, 40)  # Window sizes from 15 to 25
    
    threads = []
    
    for k in ks:
        # Create a new config file for each window size
        config = configparser.ConfigParser()
        config.read(base_config_file)
        config.set("preprocessor", "k", str(k))
        
        # Create a new output folder for each window size
        output_folder = os.path.join(base_output_folder, f"k_{k}")
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
    signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))
    import time
    while True:
        time.sleep(1)

if __name__ == "__main__":
    tune_hyperparameters_k()