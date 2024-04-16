import os
import cv2
import numpy as np
import pickle
import time
from robotcar_dataset_sdk import radar
from radar_data_loader import RadarDataLoader
from preprocessor import Preprocessor
from feature_detector import FeatureDetector
from flow_estimator import FlowEstimator
from odometry_estimator import OdometryEstimator
from visualization import Visualizer
import configparser
from odometry_evaluation import OdometryEvaluation

class Pipeline:
    def __init__(self, config):
        def str_to_bool(s):
            return s.lower() in ('true', '1', 't', 'y', 'yes')
        
        sections = ['general', 'visualization', 'preprocessor', 'feature_detector',
                    'flow_estimator', 'odometry_estimator', 'error']

        for section in sections:
            section_config = dict(config[section])
            for key, value in section_config.items():
                if isinstance(value, str) and value.lower() in ('true', 'false', 'yes', 'no', 'on', 'off'):
                    section_config[key] = str_to_bool(value)
            setattr(self, f"{section}_config", section_config)

        self.data_loader = RadarDataLoader(config=self.general_config)
        self.preprocessor = Preprocessor(config=self.preprocessor_config)
        self.feature_detector = FeatureDetector(config=self.feature_detector_config)
        self.flow_estimator = FlowEstimator(config=self.flow_estimator_config)
        self.odometry_estimator = OdometryEstimator(config=self.odometry_estimator_config)
        self.visualizer = Visualizer(config=self.visualization_config)
        self.odometry_evaluation = OdometryEvaluation(config=self.general_config)
        self.distances = [int(distance.strip()) for distance in self.error_config.get("distances", "").split(',')]
        self.step_size = int(self.error_config.get('step_size', 10))
    
    def run(self):
        tx_values, ty_values, theta_values = [], [], []

        start_time = time.time()
        timestamps_generator = self.data_loader.load_timestamps()
        pose_count = 0

        for timestamp in timestamps_generator:
            azimuth_data = self.data_loader.load_azimuth_data(radar_timestamp=timestamp)
            raw_radar_img = self.data_loader.load_cartesian_image()
            self.visualizer.update_raw_radar_img(raw_radar_img=raw_radar_img)
            processed_azimuth_data = self.preprocessor.select_strongest_returns(azimuth_data=azimuth_data)
            filtered_radar_img = self.data_loader.load_cartesian_image(processed_azimuth_data=processed_azimuth_data)
            self.visualizer.update_filtered_radar_img(filtered_radar_img=filtered_radar_img)
            features = self.feature_detector.shi_tomasi_detector(filtered_radar_img)
            self.visualizer.update_feature_point_img(feature_point_img=filtered_radar_img, features=features)
            old_points, new_points = self.flow_estimator.lk_flow(filtered_radar_img, features)
            self.visualizer.update_flow_img(flow_img=filtered_radar_img, old_points=old_points, new_points=new_points)
            tx, ty, theta = self.odometry_estimator.compute_transform(cart_pixel_width=filtered_radar_img.shape[1], old_points=old_points, new_points=new_points)
            tx_values.append(tx)
            ty_values.append(ty)
            theta_values.append(theta)
            self.odometry_evaluation.update_path(tx=tx, ty=ty, theta=theta, timestamp=timestamp)
            gt_path = self.odometry_evaluation.get_gt_path()
            pred_path = self.odometry_evaluation.get_pred_path()
            self.visualizer.update_path_plot(pred_path=pred_path, gt_path=gt_path)
            self.visualizer.update_error_plot(pred_path=pred_path, gt_path=gt_path)
            self.visualizer.show()
            pose_count += 1

        elapsed_time = time.time() - start_time
        poses_per_second = pose_count / elapsed_time if elapsed_time > 0 else 0

        # Remaining calculation and evaluations
        self.odometry_evaluation.calc_distance_interval_errors(self.distances, self.step_size)
        average_errors_by_distance, overall_avg_rotation_error, overall_avg_translation_error = self.odometry_evaluation.calc_average_errors()

        for distance, (avg_r_err, avg_t_err) in average_errors_by_distance.items():
            print(f"Average errors for {distance}m:")
            print(f"\tTranslation Error (%): {avg_t_err:.3f}")
            print(f"\tRotation Error (deg/100m): {avg_r_err:.3f}")

        print("Overall average errors:")
        print(f"Translational error (%): {overall_avg_translation_error:.3f}")
        print(f"Rotational error (deg/100m): {overall_avg_rotation_error:.3f}")

        ate_values = self.odometry_evaluation.calculate_ate_vector()
        rmse_percentage = self.odometry_evaluation.calculate_rmse_percentage()
        print(f"RMSE error: {rmse_percentage:.3f}")

        # Print the average poses processed per second
        print(f"Average poses processed per second: {poses_per_second:.2f}")

        return self.odometry_evaluation.get_pred_path(), self.odometry_evaluation.get_gt_path(), rmse_percentage, ate_values, average_errors_by_distance, overall_avg_translation_error, overall_avg_rotation_error, poses_per_second

if __name__ == "__main__":
    config_dir = r"C:\Users\SamuelChee\Desktop\FYP\opencv_optical_flow\pipeline\config\pipeline_config.ini"
    config = configparser.ConfigParser()
    config.read(config_dir)
    pipeline = Pipeline(config=config)
    pipeline.run()