import os
import cv2
import numpy as np
from robotcar_dataset_sdk import radar
from scipy.optimize import least_squares
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import torch

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
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            height, width = flow.shape[:2]

            # Create a grid of (x, y) coordinates
            y, x = np.mgrid[0:height, 0:width]

            # Shift the origin to the center
            x = x - width / 2
            y = y - height / 2

            # Get the optical flow components
            u = flow[..., 0]
            v = flow[..., 1]

            # We want to reshape the data into 1D arrays
            xs = x.reshape(-1)
            ys = y.reshape(-1)
            us = u.reshape(-1)
            vs = v.reshape(-1)


            xs = torch.tensor(xs, device='cpu').float()  # move data to GPU
            ys = torch.tensor(ys, device='cpu').float()
            us = torch.tensor(us, device='cpu').float()
            vs = torch.tensor(vs, device='cpu').float()
            num_points = len(xs)

            # Create A as a (2*num_points, 6) tensor
            A = torch.zeros(2*num_points, 6)
            A[:num_points, 0] = xs
            A[:num_points, 1] = ys
            A[:num_points, 2] = 1
            A[num_points:, 3] = xs
            A[num_points:, 4] = ys
            A[num_points:, 5] = 1

           # Create b as a (2*num_points, 1) tensor
            b = torch.zeros(2*num_points, 1)
            b[:num_points, 0] = us + xs
            b[num_points:, 0] = vs + ys

            # Use torch.lstsq to solve for the affine transformation parameters
            solution = torch.linalg.lstsq(A, b)

            a, b, c, d, tx, ty = solution.solution[:6, 0].cpu().numpy()
            # Calculate the rotation angle in radians
            theta_rad = np.arctan2(b, a)

            tx_values.append(tx)
            ty_values.append(ty)
            theta_values.append(theta_rad)


            
        prev_frame = cart_img.copy()

    data = {"tx": tx_values, "ty": ty_values, "theta": theta_values}
    # Save the dictionary to a pickle file
    with open('data_gpu.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
