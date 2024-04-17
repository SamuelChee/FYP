import cv2
import numpy as np

class Preprocessor:

    def __init__(self, config):
        self.k = int(config["k"])
        self.z_min = float(config["z_min"]) 

    def select_strongest_returns(self, azimuth_data):

        
        assert self.k > 0 and self.z_min >= 0, "k must be > 0 and z_min must be >= 0"
        
        # Flatten azimuth_data to 2D (400, 3768) for easier processing
        azimuth_data_2d = azimuth_data.reshape(azimuth_data.shape[0], -1)

        # Sort by power (each row of azimuth_data represents power for an azimuth)
        sorted_indices = np.argsort(azimuth_data_2d, axis=-1)[:, ::-1]

        # Select top k returns
        top_k_indices = sorted_indices[:, :self.k]

        # Create a mask where power exceeds z_min
        mask = np.take_along_axis(azimuth_data_2d, top_k_indices, axis=-1) >= self.z_min

        # Use the mask to make sure values below z_min are 0
        selected_power = np.where(mask, np.take_along_axis(azimuth_data_2d, top_k_indices, axis=-1), 0)

        # Initialize a new array with zeros
        selected_azimuth_data = np.zeros_like(azimuth_data_2d)

        # Create a 2D array of indices for the rows
        row_indices = np.arange(azimuth_data_2d.shape[0])[:, None]

        # Place the selected returns in their original positions using advanced indexing
        selected_azimuth_data[row_indices, top_k_indices] = selected_power

        # Reshape the array back to the original shape
        selected_azimuth_data = selected_azimuth_data.reshape(azimuth_data.shape)

        return selected_azimuth_data
    










