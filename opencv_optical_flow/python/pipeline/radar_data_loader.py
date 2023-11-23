import os
import cv2
import numpy as np
from robotcar_dataset_sdk import radar


class RadarDataLoader:
    def __init__(self, radar_dir, cart_resolution=0.1, cart_pixel_width=500, interpolate_crossover=True):
        self.radar_dir = radar_dir
        self.cart_resolution = cart_resolution
        self.cart_pixel_width = cart_pixel_width
        self.interpolate_crossover = interpolate_crossover
        self.timestamps = None
        self.azimuths = None
        self.valid = None
        self.azimuth_data = None
        self.radar_resolution = None

    def load_timestamps(self):
        timestamps_path = os.path.join(
            self.radar_dir, os.pardir, "radar.timestamps")
        if not os.path.isfile(timestamps_path):
            raise IOError("Could not find timestamps file")

        radar_timestamps = np.loadtxt(
            timestamps_path, delimiter=" ", usecols=[0], dtype=np.int64
        )

        for timestamp in radar_timestamps:
            yield timestamp
    
    def load_azimuth_data(self, radar_timestamp):
        filename = os.path.join(self.radar_dir, str(radar_timestamp) + ".png")
        if not os.path.isfile(filename):
            raise FileNotFoundError("Could not find radar example: {}".format(filename))

        self.timestamps, self.azimuths, self.valid, self.azimuth_data, self.radar_resolution = radar.load_radar(filename)
        return self.azimuth_data

    def load_cartesian_image(self, processed_azimuth_data=None):
        assert self.azimuths is not None and self.azimuth_data is not None and self.radar_resolution is not None, "You must call load_azimuths() first"
        if(processed_azimuth_data is not None):
            self.azimuth_data = processed_azimuth_data

        cart_img = radar.radar_polar_to_cartesian(
            self.azimuths,
            self.azimuth_data,
            self.radar_resolution,
            self.cart_resolution,
            self.cart_pixel_width,
            self.interpolate_crossover,
        )
        cart_img = cv2.convertScaleAbs(cart_img * 255.0)
        return cart_img

