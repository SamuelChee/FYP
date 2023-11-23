import cv2
import numpy as np

class FlowEstimator:
    def __init__(self):
        self.prev_gray = None
        self.prev_features = None

    def lk_flow(self, img, features):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        if self.prev_gray is not None and self.prev_features is not None:
            next_features, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_features, None)
            valid_old = self.prev_features[status == 1]
            valid_new = next_features[status == 1]
        else:
            valid_old = valid_new = np.array([])

        self.prev_gray = gray
        self.prev_features = features

        return valid_old, valid_new