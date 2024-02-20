import cv2
import numpy as np
class FeatureDetector:

    def __init__(self, config):
        self.max_features = int(config["max_features"])
        self.quality_level = float(config["quality_level"])
        self.min_distance = float(config["min_distance"]) 
    """
    A feature detector that uses the Shi-Tomasi corner detection method.
    """
    
    def shi_tomasi_detector(self, img):
        """
        Detects Shi-Tomasi features in the given image.

        Parameters:
        img (numpy.ndarray): The input image. Can be a grayscale or a color image.
        max_features (int, optional): The maximum number of features to return. If there are more features
            in the image, the strongest ones will be returned. Default is 100.
        quality_level (float, optional): Parameter characterizing the minimal accepted quality of image features.
            The parameter value is multiplied by the best corner quality measure (smallest eigenvalue).
            The features with the quality measure less than the product are rejected. For example, if the best
            corner has the quality measure = 1500, and the quality_level = 0.01, then all the features with the
            quality measure less than 15 are rejected. Default is 0.01.
        min_distance (float, optional): Minimum possible Euclidean distance between the returned features.
            Default is 10.

        Returns:
        features (numpy.ndarray): Output array of detected features, each represented as [x, y].
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        features = cv2.goodFeaturesToTrack(gray, self.max_features, self.quality_level, self.min_distance)
        return features