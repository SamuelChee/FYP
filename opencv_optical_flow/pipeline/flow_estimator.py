import cv2
import numpy as np

class FlowEstimator:
    def __init__(self, config):
        self.prev_gray = None
        self.prev_features = None
        self.win_size = (int(config["window_size"]), int(config["window_size"]))
        self.max_level = int(config["max_level"])
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(config["criteria_max_count"]), float(config["criteria_epsilon"]) )
        self.flags = int(config["flags"])
        self.min_eig_threshold = float(config["min_eig_threshold"])

        

    def lk_flow(self, img, features):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        if self.prev_gray is not None and self.prev_features is not None:
            next_features, status, err = cv2.calcOpticalFlowPyrLK(prevImg=self.prev_gray,
                                                                  nextImg=gray,
                                                                  prevPts=self.prev_features,
                                                                  nextPts=None, # nextPts will be computed
                                                                  winSize= self.win_size,
                                                                  maxLevel=self.max_level,
                                                                  criteria=self.criteria,
                                                                  flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                                                                  minEigThreshold=self.min_eig_threshold)
            valid_old = self.prev_features[status == 1]
            valid_new = next_features[status == 1]
        else:
            valid_old = valid_new = np.array([])

        self.prev_gray = gray
        self.prev_features = features

        return valid_old, valid_new