import os
import cv2
import numpy as np
from robotcar_dataset_sdk import radar
import pandas as pd

def main():
    radar_dir = r"C:\Users\SamuelChee\Desktop\FYP\data\2019-01-10-14-36-48-radar-oxford-10k-partial-large\radar"
    timestamps_path = os.path.join(
        os.path.join(radar_dir, os.pardir, "radar.timestamps")
    )
    if not os.path.isfile(timestamps_path):
        raise IOError("Could not find timestamps file")

    # Cartesian Visualization Setup
    cart_resolution = (
        0.1  # Resolution of the cartesian form of the radar scan in meters per pixel
    )
    cart_pixel_width = (
        500  # Cartesian visualization size (used for both height and width)
    )
    interpolate_crossover = True
    title = "Radar Visualization Example"

    radar_timestamps = np.loadtxt(
        timestamps_path, delimiter=" ", usecols=[0], dtype=np.int64
    )
    try:
        cap = cv2.VideoCapture(0)

    except cv2.error as e:
        print("OpenCV Error:", e)
        # Handle the error here, such as displaying an error message or logging it
    except Exception as e:
        print("Error:", e)
        # Handle any other exceptions that may occur

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Failed to open webcam.")
        # Handle the failure here, such as displaying an error message or exiting the program


    prev_frame = None

    # Load the CSV data
    odometry_data = pd.read_csv(r'C:\Users\SamuelChee\Desktop\FYP\data\2019-01-10-14-36-48-radar-oxford-10k-partial-large\gt\radar_odometry.csv')

    # Create a dictionary mapping timestamps to x, y coordinates
    timestamp_to_coordinates = {
        row['source_radar_timestamp']: (row['x'], row['y'])
        for _, row in odometry_data.iterrows()
    }

    # Initialize a list to store past positions
    past_positions = []



    for radar_timestamp in radar_timestamps:
        filename = os.path.join(radar_dir, str(radar_timestamp) + ".png")

        if not os.path.isfile(filename):
            raise FileNotFoundError("Could not find radar example: {}".format(filename))

        timestamps, azimuths, valid, fft_data, radar_resolution = radar.load_radar(
            filename
        )

        cart_img = radar.radar_polar_to_cartesian(
            azimuths,
            fft_data,
            radar_resolution,
            cart_resolution,
            cart_pixel_width,
            interpolate_crossover,
        )
        
        cart_img = cv2.convertScaleAbs(cart_img * 255.0)
        
        cart_img = cv2.cvtColor(cart_img, cv2.COLOR_GRAY2BGR)


        # If the timestamp of this image is in the dictionary, add the corresponding coordinates to the list of past positions
        if radar_timestamp in timestamp_to_coordinates:
            print(radar_timestamp)
            # Convert real-world coordinates to pixel coordinates
            x, y = timestamp_to_coordinates[radar_timestamp]
            pixel_x = int(x / cart_resolution)
            pixel_y = int(y / cart_resolution)

            # Add the current position to the list of past positions
            past_positions.append((pixel_x, pixel_y))

        # Calculate the shift needed to keep the newest pose point at the center of the image
        center_x, center_y = past_positions[-1]
        shift_x = cart_pixel_width // 2 - center_x
        shift_y = cart_pixel_width // 2 - center_y

        # Draw all past positions onto the image, shifted appropriately
        # print(past_positions)
        for pos_x, pos_y in past_positions:
            cv2.circle(cart_img, (pos_x + shift_x, pos_y + shift_y), 2, (0, 0, 255), -1)

        if prev_frame is not None:
            cv2.imshow("cart_img", cart_img)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        prev_frame = cart_img.copy()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
