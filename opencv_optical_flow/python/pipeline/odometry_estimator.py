import numpy as np
import cv2


class OdometryEstimator:
    def __init__(self):
        self.init_flag = False

    def compute_transform(self, cart_pixel_width, old_points, new_points):
        if self.init_flag:
            # Convert points to floating point precision
            old_points = old_points.astype(np.float32)
            new_points = new_points.astype(np.float32)

            # Get image dimensions (assuming old_points and new_points come from the same-sized image)
            img_height, img_width = cart_pixel_width, cart_pixel_width

            # Shift points to change origin from top-left to center
            old_points_shifted = old_points - \
                np.array([img_width / 2, img_height / 2])
            new_points_shifted = new_points - \
                np.array([img_width / 2, img_height / 2])

            # Estimate the affine transform
            # Try rigid transform
            affine_transform, inliers = cv2.estimateAffinePartial2D(
                old_points_shifted,
                new_points_shifted,
                method=cv2.RANSAC,
                ransacReprojThreshold=2,
                confidence=0.99,
                maxIters=2000,
            )

            if affine_transform is None:
                return None, None, None

            # Extract translation from the affine transform
            dx, dy = affine_transform[0, 2], affine_transform[1, 2]

            # Extract rotation from the affine transform
            a, b = affine_transform[0, 0], affine_transform[1, 0]
            theta = np.arctan2(b, a) * 180 / np.pi

            return dx, dy, theta
        self.init_flag = True
        return None, None, None
