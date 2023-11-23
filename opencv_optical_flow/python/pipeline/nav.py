import numpy as np

class SE2Pose:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
    
    def __str__(self):
        return f"SE2Pose: x={self.x}, y={self.y}, theta={self.theta}"

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

    def add_relative_pose(self, relative_pose, timestamp):
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
    
