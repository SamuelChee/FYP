import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import os
import pickle
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

# Initialize arrays to store the global x and y positions
fwd_x_global = np.zeros(len(radar_timestamps))
rwd_y_global = np.zeros(len(radar_timestamps))

cart_resolution = (
    0.1  # Resolution of the cartesian form of the radar scan in meters per pixel
)
# Initialize the current yaw angle
current_yaw = 0.0


# Open the file in read-binary mode using 'rb'
with open("data.pickle", "rb") as file:
    pred_data = pickle.load(file)

print(pred_data)

len_pred = len(pred_data["tx"])

fwd_x_global_pred = np.zeros(len_pred)
rwd_y_global_pred = np.zeros(len_pred)
current_yaw_pred = 0.0

for i in range(len_pred):
    x =  pred_data["tx"][i] * cart_resolution 
    y =   pred_data["ty"][i] * cart_resolution
    yaw = - ((pred_data["theta"][i]) * (np.pi / 180.0))

    current_yaw_pred += yaw

    fwd_dx_global_pred = x*np.cos(current_yaw_pred) - y*np.sin(current_yaw_pred)
    rwd_dy_global_pred = x*np.sin(current_yaw_pred) + y*np.cos(current_yaw_pred)

    # Update the global positions
    fwd_x_global_pred[i] = fwd_x_global_pred[i-1] + fwd_dx_global_pred
    rwd_y_global_pred[i] = rwd_y_global_pred[i-1] + rwd_dy_global_pred


for i, radar_timestamp in enumerate(radar_timestamps):
    if radar_timestamp in timestamp_to_coordinates:
        x, y, yaw = timestamp_to_coordinates[radar_timestamp]
        # Update the yaw angle
        current_yaw += yaw

        # Compute the changes in position in the global coordinate frame
        fwd_dx_global = x*np.cos(current_yaw) - y*np.sin(current_yaw)
        rwd_dy_global = x*np.sin(current_yaw) + y*np.cos(current_yaw)

        # Update the global positions
        fwd_x_global[i] = fwd_x_global[i-1] + fwd_dx_global
        rwd_y_global[i] = rwd_y_global[i-1] + rwd_dy_global

    else:
        continue

fig, ax = plt.subplots(figsize=(8, 8))

# Set the aspect of the plot to be equal, so the path is shown correctly
ax.set_aspect('equal', adjustable='box')

# Add labels and title
ax.set_xlabel('rwd position (m)')
ax.set_ylabel('Forward Position (m)')
ax.set_title('Path of the SE2 poses')

gt_line, = ax.plot([], [])
pred_line, = ax.plot([], [], color='red')  # Create an additional line for predictions

ax.set_xlim(fwd_x_global.min(), fwd_x_global.max())
ax.set_ylim(rwd_y_global.min(), rwd_y_global.max())
    
def animate(i):
    # Swap the x and y coordinates
    gt_line.set_data(rwd_y_global[:i], fwd_x_global[:i])
    # Update the prediction line data
    pred_line.set_data(rwd_y_global_pred[:i], fwd_x_global_pred[:i])

    return gt_line, pred_line,  # Return both lines

ani = FuncAnimation(fig, animate, frames=len(fwd_x_global), interval=30, blit=True, repeat=False)

plt.show()