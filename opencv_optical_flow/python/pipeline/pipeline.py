import os
import cv2
import numpy as np
import pickle
from robotcar_dataset_sdk import radar
from radar_data_loader import RadarDataLoader
from preprocessor import Preprocessor
from feature_detector import FeatureDetector
from flow_estimator import FlowEstimator
from odometry_estimator import OdometryEstimator
from visualization import Visualizer
import configparser

class Pipeline:
    def __init__(self, config):
        def str_to_bool(s):
            return s.lower() in ('true', '1', 't', 'y', 'yes')
        sections = [
            'save', 'visualization', 'radar_loader',
            'preprocessor', 'feature_detector',
            'optical_flow_pyr_lk', 'odometry_estimator'
        ]
        for section in sections:
            # Retrieve the section as a dictionary
            section_config = dict(config[section])
            # Iterate through the items in the section and convert any boolean strings to bools
            for key, value in section_config.items():
                if isinstance(value, str):
                    # Check if the value is a boolean string and convert if so
                    if value.lower() in ('true', 'false', 'yes', 'no', 'on', 'off'):
                        section_config[key] = str_to_bool(value)
            # Set the converted configuration section on the object
            setattr(self, f"{section}_config", section_config)

        self.data_loader = RadarDataLoader(config=self.radar_loader_config)
        self.preprocessor = Preprocessor()
        self.feature_detector = FeatureDetector()
        self.flow_estimator = FlowEstimator()
        self.odometry_estimator = OdometryEstimator()
        self.visualizer = Visualizer(config=self.visualization_config)

        
    
    def run(self):

        tx_values = []
        ty_values = []
        theta_values = []

        timestamps_generator = self.data_loader.load_timestamps()
        for timestamp in timestamps_generator:
            azimuth_data = self.data_loader.load_azimuth_data(radar_timestamp=timestamp)
            raw_radar_img = self.data_loader.load_cartesian_image()
            self.visualizer.update(raw_radar_img = raw_radar_img)
            processed_azimuth_data = self.preprocessor.select_strongest_returns(azimuth_data=azimuth_data, k=15, z_min=0.35)
            filtered_radar_img = self.data_loader.load_cartesian_image(processed_azimuth_data=processed_azimuth_data)
            self.visualizer.update(filtered_radar_img = filtered_radar_img)
            #Todo: ry bypass detector
            features = self.feature_detector.shi_tomasi_detector(filtered_radar_img, max_features=200, quality_level=0.01, min_distance=20)
            self.visualizer.update(feature_point_img = filtered_radar_img, features = features)
            old_points, new_points = self.flow_estimator.lk_flow(filtered_radar_img, features)
            self.visualizer.update(flow_img = filtered_radar_img, old_points = old_points, new_points = new_points)
            tx, ty, theta = self.odometry_estimator.compute_transform(cart_pixel_width=filtered_radar_img.shape[1], old_points=old_points, new_points=new_points)

            print("tx: ", tx, "ty: ", ty, "theta: ", theta)
            tx_values.append(tx)
            ty_values.append(ty)
            theta_values.append(theta)


            self.visualizer.show()
        

        # data = {"tx": tx_values, "ty": ty_values, "theta": theta_values}
        # with open('data.pickle', 'wb') as handle:
        #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


        self.visualizer.close()
        cv2.destroyAllWindows()

            
  

if __name__ == "__main__":
    config_dir = r"C:\Users\SamuelChee\Desktop\FYP\opencv_optical_flow\python\pipeline\config\pipeline_config.ini"
    config = configparser.ConfigParser()
    config.read(config_dir)
    pipeline = Pipeline(config=config)
    pipeline.run()