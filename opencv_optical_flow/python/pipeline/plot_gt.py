import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import os
import pickle
import nav
radar_dir = r"C:\Users\SamuelChee\Desktop\FYP\data\2019-01-10-14-36-48-radar-oxford-10k-partial-large\radar"
timestamps_path = os.path.join(
    os.path.join(radar_dir, os.pardir, "radar.timestamps")
)

if not os.path.isfile(timestamps_path):
        raise IOError("Could not find timestamps file")

radar_timestamps = np.loadtxt(
        timestamps_path, delimiter=" ", usecols=[0], dtype=np.int64
    )

# Load the CSV data
gt_data = pd.read_csv(r'C:\Users\SamuelChee\Desktop\FYP\data\2019-01-10-14-36-48-radar-oxford-10k-partial-large\gt\radar_odometry.csv')
 # Create a dictionary mapping timestamps to x, y coordinates
timestamp_to_coordinates = {
    row['source_radar_timestamp']: (row['x'], row['y'], row['yaw'])
    for _, row in gt_data.iterrows()
}

gt_path = nav.Path()
pred_path = nav.Path()

cart_resolution = (
    0.1  # Resolution of the cartesian form of the radar scan in meters per pixel
)

# Open the file in read-binary mode using 'rb'
with open(r"C:\Users\SamuelChee\Desktop\FYP\opencv_optical_flow\path\outlier.pickle", "rb") as file:
    pred_data = pickle.load(file)


for i in range(len(pred_data["tx"])):
    if pred_data["tx"][i] is not None:
        dx =  pred_data["tx"][i] * cart_resolution 
        dy =   pred_data["ty"][i] * cart_resolution
        dtheta = - ((pred_data["theta"][i]) * (np.pi / 180.0))
        pred_path.add_relative_pose(nav.SE2Pose(dx, dy, dtheta), radar_timestamps[i])

print(pred_path)
pred_path.apply_transform(transform=nav.SE2Transform(theta=-np.pi/2))
for i, radar_timestamp in enumerate(radar_timestamps):
    if radar_timestamp in timestamp_to_coordinates:
        dx, dy, dtheta = timestamp_to_coordinates[radar_timestamp]
        gt_path.add_relative_pose(nav.SE2Pose(dx, dy, dtheta), radar_timestamp)


fig, ax = plt.subplots(figsize=(8, 8))

# Set the aspect of the plot to be equal, so the path is shown correctly
ax.set_aspect('equal', adjustable='box')

# Add labels and title
ax.set_xlabel('rwd position (m)')
ax.set_ylabel('Forward Position (m)')
ax.set_title('Path of the SE2 poses')

gt_line, = ax.plot([], [])
pred_line, = ax.plot([], [], color='red')  # Create an additional line for predictions


ax.set_xlim(-gt_path.max_x()-50, gt_path.max_x()+50)
ax.set_ylim(-gt_path.max_x()-50, gt_path.max_x()+50)
    
def animate(i):
    # Swap the x and y coordinates
    gt_line.set_data(gt_path.y_coords()[:i], gt_path.x_coords()[:i])
    # Update the prediction line data
    pred_line.set_data(pred_path.y_coords()[:i], pred_path.x_coords()[:i])

    return gt_line, pred_line,  # Return both lines

ani = FuncAnimation(fig, animate, frames=len(gt_path), interval=30, blit=True, repeat=False)

plt.show()