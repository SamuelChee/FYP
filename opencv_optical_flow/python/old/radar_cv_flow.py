import os
import cv2
import numpy as np
from robotcar_dataset_sdk import radar
from scipy.optimize import least_squares
import pickle

def draw_flow(img, flow, step=64):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def main():
    radar_dir = r"C:\Users\SamuelChee\Desktop\FYP\data\2019-01-10-14-36-48-radar-oxford-10k-partial-large\radar"
    timestamps_path = os.path.join(
        os.path.join(radar_dir, os.pardir, "radar.timestamps")
    )
    print(timestamps_path)
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
 
    # Initialize empty lists to store tx, ty, and theta values
    tx_values = []
    ty_values = []
    theta_values = []

    prev_frame = None
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

        if prev_frame is not None:
            cv2.imshow("cart_img", cart_img)
            print(np.min(cart_img), np.max(cart_img))
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, cart_img, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_img = draw_flow(cart_img, flow)
            cv2.imshow("flow", flow_img)


            x, y = np.meshgrid(np.arange(flow.shape[1]), np.arange(flow.shape[0]))
            coords = np.stack([x, y], axis=-1)

            # Reshape arrays to 1D
            flow = flow.reshape(-1, 2)
            coords = coords.reshape(-1, 2)

            # Function to compute residuals
            def residuals(params, coords, flow):
                a, b, tx, c, d, ty = params
                # Adjust the coordinates to consider the center of the image as the origin
                coords_centered = coords - [cart_pixel_width / 2, cart_pixel_width / 2]
                predicted_positions = np.dot(coords_centered, [[a, b], [c, d]]) + [tx, ty]
                predicted_flow = predicted_positions - coords_centered
                return (predicted_flow - flow).ravel()
            
            # Function to suppress outliers in the optical flow
            def suppress_outliers(flow, coords, params):
                # Calculate residuals
                residuals_val = residuals(params, coords, flow)

                # Reshape residuals back to 2D
                residuals_val = residuals_val.reshape(-1, 2)

                # Calculate the magnitude of the residuals
                residuals_mag = np.sqrt(residuals_val[..., 0] ** 2 + residuals_val[..., 1] ** 2)

                # Choose a threshold
                threshold = np.percentile(residuals_mag, 90)  # for example, the 90th percentile

                # Create a mask of the inliers
                inlier_mask = residuals_mag < threshold

                # Suppress the outliers in the optical flow
                return flow[inlier_mask], coords[inlier_mask]

            # Initial guess (no rotation, no scaling, no shearing, no translation)
            params0 = [1, 0, 0, 0, 1, 0]

            # Solve for the parameters
            res = least_squares(residuals, params0, args=(coords, flow))
            params = res.x

            # Suppress outliers in the optical flow
            flow, coords = suppress_outliers(flow, coords, params)

            # Solve for the parameters
            res = least_squares(residuals, params, args=(coords, flow))
            a, b, tx, c, d, ty = res.x

            theta = np.arctan2(b, a) * 180 / np.pi


            print("X:  ", tx, "  Y:  ", ty, "  T:  ", theta)

            tx_values.append(tx)
            ty_values.append(ty)
            theta_values.append(theta)


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        prev_frame = cart_img.copy()

    data = {"tx": tx_values, "ty": ty_values, "theta": theta_values}
    # Save the dictionary to a pickle file
    with open('outlier.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
