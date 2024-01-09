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
old_path = nav.Path()

cart_resolution = (
    0.1  # Resolution of the cartesian form of the radar scan in meters per pixel
)

# Open the file in read-binary mode using 'rb'
with open(r"C:\Users\SamuelChee\Desktop\FYP\opencv_optical_flow\python\pipeline\data_okay.pickle", "rb") as file:
    pred_data = pickle.load(file)

    # Open the file in read-binary mode using 'rb'
with open(r"C:\Users\SamuelChee\Desktop\FYP\opencv_optical_flow\results\old_results.pickle", "rb") as file:
    old_data = pickle.load(file)

for i in range(len(old_data["tx"])):
    if old_data["tx"][i] is not None:
        dx =  old_data["tx"][i] * cart_resolution 
        dy =   old_data["ty"][i] * cart_resolution
        dtheta = - ((old_data["theta"][i]) * (np.pi / 180.0))
        old_path.add_relative_pose(nav.SE2Pose(dx, dy, dtheta), radar_timestamps[i])

old_path.apply_transform(transform=nav.SE2Transform(theta=-np.pi/2))

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

gt_line, = ax.plot([], [], label='Ground Truth')
pred_line, = ax.plot([], [], color='red', label='With Data Processing')  # Create an additional line for predictions
old_line, = ax.plot([], [], color='green', label='No Data Processing')  # Create an additional line for predictions



ax.set_xlim(-350, 120)
ax.set_ylim(-50, 350)
    
def animate(i):
    # Swap the x and y coordinates
    gt_line.set_data(gt_path.y_coords()[:i], gt_path.x_coords()[:i])
    # Update the prediction line data
    pred_line.set_data(pred_path.y_coords()[:i], pred_path.x_coords()[:i])

    old_line.set_data(old_path.y_coords()[:i], old_path.x_coords()[:i])

    return gt_line, pred_line,  old_line,# Return both lines

ani = FuncAnimation(fig, animate, frames=len(gt_path), interval=30, blit=True, repeat=False)
ax.legend(loc='upper left')
plt.show()