import os
import cv2
import numpy as np
from robotcar_dataset_sdk import radar


def draw_flow(img, flow, step=16):
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
    radar_dir = "data/oxford-radar-robotcar-dataset/2019-01-10-14-36-48-radar-oxford-10k-partial/radar"

    timestamps_path = os.path.join(
        os.path.join(radar_dir, os.pardir, "radar.timestamps")
    )
    if not os.path.isfile(timestamps_path):
        raise IOError("Could not find timestamps file")

    # Cartesian Visualization Setup
    cart_resolution = (
        0.2  # Resolution of the cartesian form of the radar scan in meters per pixel
    )
    cart_pixel_width = (
        1000  # Cartesian visualization size (used for both height and width)
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

    # Parameters for optical flow calculation
    lk_params = dict(
        winSize=(10, 10),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

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
        # ret, frame = cap.read()
        # cart_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            cv2.imshow("prev_frame", prev_frame)
            cv2.imshow("cart_img", cart_img)
            print(np.min(prev_frame), np.max(prev_frame))
            print(np.min(cart_img), np.max(cart_img))
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, cart_img, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            flow_img = draw_flow(cart_img, flow)
            cv2.imshow("flow", flow_img)
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        prev_frame = cart_img.copy()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
