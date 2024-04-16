import configparser
import os
import shutil
from pipeline import Pipeline
import numpy as np
import pickle

def run_pipeline(config_file, output_folder):
    config = configparser.ConfigParser()
    config.read(config_file)
    pipeline = Pipeline(config=config)
    
    pred_path, gt_path, average_errors_by_distance, overall_avg_translation_error, overall_avg_rotation_error, poses_per_second = pipeline.run()
    
    # Save the error values in a text file
    with open(os.path.join(output_folder, "error_values.txt"), "w") as f:
        f.write("Average errors by distance:\n")
        for distance, (avg_r_err, avg_t_err) in average_errors_by_distance.items():
            f.write(f"Average errors for {distance}m:\n")
            f.write(f"\tTranslation Error (%): {avg_t_err * 100:.3f}\n")
            f.write(f"\tRotation Error (deg/100m): {avg_r_err / np.pi * 180 * 100:.3f}\n")
        
        f.write("\nOverall average errors:\n")
        f.write(f"Translational error (%): {overall_avg_translation_error * 100:.3f}\n")
        f.write(f"Rotational error (deg/100m): {overall_avg_rotation_error * (180 / np.pi) * 100:.3f}\n")

        f.write("\nRuntime:\n")
        f.write(f"Poses per second: {poses_per_second:.3f}\n")


    
    # Save the tx, ty, theta values in a pickle file
    data = {"pred_path": pred_path, "gt_path": gt_path}
    with open(os.path.join(output_folder, "data.pickle"), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save the last frame Matplotlib figure
    pipeline.visualizer.save_figure(os.path.join(output_folder, "last_frame_figure.png"))
    pipeline.visualizer.close()
    

def tune_hyperparameters():
    base_config_file = "config/pipeline_config.ini"
    base_output_folder = "tuning_results"
    
    window_sizes = range(20, 26)  # Window sizes from 20 to 25
    
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
        
        # Run the pipeline sequentially
        run_pipeline(config_file, output_folder)

if __name__ == "__main__":
    tune_hyperparameters()