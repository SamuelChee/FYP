import numpy as np

class SE2Pose:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
    
    def __str__(self):
        return f"SE2Pose: x={self.x}, y={self.y}, theta={self.theta}"

    def distance_to(self, other_pose):
        """Calculate the Euclidean distance between this pose and another pose."""
        return np.sqrt((self.x - other_pose.x) ** 2 + (self.y - other_pose.y) ** 2)


class SE2Transform:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
    
    def __mul__(self, pose):
        # Apply the SE2 transform to a pose
        dx = self.x + pose.x * np.cos(self.theta) - pose.y * np.sin(self.theta)
        dy = self.y + pose.x * np.sin(self.theta) + pose.y * np.cos(self.theta)
        dtheta = self.theta + pose.theta
        return SE2Pose(dx, dy, dtheta)
    
class Path:
    def __init__(self):
        self.poses = []
        self.timestamps = []
    
    def add_pose(self, pose, timestamp):
        self.poses.append(pose)
        self.timestamps.append(timestamp)

    def apply_transform(self, transform):
        for i in range(len(self.poses)):
            pose = self.poses[i]
            transformed_pose = transform * pose  # Apply the SE2 transform
            self.poses[i] = transformed_pose

    def add_relative_pose(self, relative_pose, timestamp=0.0):
        dx_rel = relative_pose.x
        dy_rel = relative_pose.y
        dtheta = relative_pose.theta

        if len(self.poses) == 0:
            cur_theta = dtheta
            dx = dx_rel * np.cos(cur_theta) - dy_rel * np.sin(cur_theta)
            dy = dx_rel * np.sin(cur_theta) + dy_rel * np.cos(cur_theta)
            self.add_pose(SE2Pose(dx, dy, cur_theta), timestamp)
        else:
            cur_theta = self.poses[-1].theta + dtheta
            dx = dx_rel * np.cos(cur_theta) - dy_rel * np.sin(cur_theta)
            dy = dx_rel * np.sin(cur_theta) + dy_rel * np.cos(cur_theta)
            self.add_pose(SE2Pose(self.poses[-1].x + dx, self.poses[-1].y + dy, cur_theta), timestamp)

        # def calculate_distance(pose1, pose2):
        #     return np.sqrt((pose1.x - pose2.x) ** 2 + (pose1.y - pose2.y) ** 2)

        # def calculate_ate_per_100m(gt_path, est_path):
        #     assert len(gt_path.poses) == len(est_path.poses), "The number of poses must be the same in both paths."

        #     segment_ate_values = []
        #     distance_traveled = 0.0
        #     segment_errors = []

        #     for i in range(1, len(gt_path.poses)):
        #         # Calculate the distance between consecutive ground truth poses
        #         distance = calculate_distance(gt_path.poses[i-1], gt_path.poses[i])
        #         distance_traveled += distance

        #         # Calculate the error for the current pose
        #         error = calculate_distance(gt_path.poses[i], est_path.poses[i])
        #         segment_errors.append(error)

        #         # When the segment reaches 100 meters, calculate the ATE for this segment
        #         if distance_traveled >= 100.0 or i == len(gt_path.poses) - 1:
        #             # Compute the ATE for the segment
        #             segment_ate = np.sqrt(np.mean(np.square(segment_errors)))
        #             segment_ate_values.append(segment_ate)
        #             # Reset for the next segment
        #             distance_traveled = 0.0
        #             segment_errors = []

        #     return segment_ate_values

    def calculate_ate_vector(self, other_path):
        """
        Calculate the ATE vector, assuming both paths are of the same length.
        
        :param other_path: The other Path object to compare with
        :return: List of ATE values for each corresponding pose
        """
        assert len(self.poses) == len(other_path.poses), "Paths are not of the same length."

        ate_vector = []
        for pose_self, pose_other in zip(self.poses, other_path.poses):
            ate = pose_self.distance_to(pose_other)
            ate_vector.append(ate)
        
        return ate_vector
    
    def get_pose_at_timestamp(self, timestamp):
        # Find the index of the closest timestamp
        index = min(range(len(self.timestamps)), key=lambda i: abs(self.timestamps[i] - timestamp))
        
        return self.poses[index]
    
    def pose_exists_at_timestamp(self, timestamp):
        return timestamp in self.timestamps
    
    def min_x(self):
        if not self.poses:
            return None
        return min(self.poses, key=lambda pose: pose.x).x
    
    def max_x(self):
        if not self.poses:
            return None
        return max(self.poses, key=lambda pose: pose.x).x
    
    def min_y(self):
        if not self.poses:
            return None
        return min(self.poses, key=lambda pose: pose.y).y
    
    def max_y(self):
        if not self.poses:
            return None
        return max(self.poses, key=lambda pose: pose.y).y
    
    def x_coords(self):

        return [pose.x for pose in self.poses]
    
    def y_coords(self):

        return [pose.y for pose in self.poses]
    
    def __len__(self):
        return len(self.poses)
    
    def __str__(self):
        path_str = "Path:\n"
        for i in range(len(self.poses)):
            pose = self.poses[i]
            timestamp = self.timestamps[i]
            path_str += f"Timestamp: {timestamp}, Pose: {pose}\n"
        return path_str
    
