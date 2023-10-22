# import os
# import cv2
# import numpy as np
# from robotcar_dataset_sdk import radar
# from scipy.optimize import least_squares
# import pickle
# def draw_flow(img, flow, step=64):
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
#     fx, fy = flow[y, x].T
#     lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines + 0.5)
#     vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     cv2.polylines(vis, lines, 0, (0, 255, 0))
#     for (x1, y1), (x2, y2) in lines:
#         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
#     return vis


# def main():
#     radar_dir = r"C:\Users\SamuelChee\Desktop\FYP\data\2019-01-10-14-36-48-radar-oxford-10k-partial-large\radar"
#     timestamps_path = os.path.join(
#         os.path.join(radar_dir, os.pardir, "radar.timestamps")
#     )
#     print(timestamps_path)
#     if not os.path.isfile(timestamps_path):
#         raise IOError("Could not find timestamps file")

#     # Cartesian Visualization Setup
#     cart_resolution = (
#         0.1  # Resolution of the cartesian form of the radar scan in meters per pixel
#     )
#     cart_pixel_width = (
#         500  # Cartesian visualization size (used for both height and width)
#     )
#     interpolate_crossover = True
#     title = "Radar Visualization Example"

#     radar_timestamps = np.loadtxt(
#         timestamps_path, delimiter=" ", usecols=[0], dtype=np.int64
#     )

#     # parameters for ShiTomasi corner detection
#     feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

#     # parameters for Lucas Kanade optical flow
#     lk_params = dict(winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#     prev_frame = None
#     for radar_timestamp in radar_timestamps:
#         filename = os.path.join(radar_dir, str(radar_timestamp) + ".png")

#         if not os.path.isfile(filename):
#             raise FileNotFoundError("Could not find radar example: {}".format(filename))

#         timestamps, azimuths, valid, fft_data, radar_resolution = radar.load_radar(
#             filename
#         )

#         cart_img = radar.radar_polar_to_cartesian(
#             azimuths,
#             fft_data,
#             radar_resolution,
#             cart_resolution,
#             cart_pixel_width,
#             interpolate_crossover,
#         )
        
#         if prev_frame is not None:
#             # convert to grayscale

#             # find corners in the previous frame
#             p0 = cv2.goodFeaturesToTrack(prev_frame, mask = None, **feature_params)

#             # if corners were found
#             if p0 is not None:
#                 # calculate optical flow
#                 p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, cart_img, p0, None, **lk_params)

#                 # select good points
#                 good_new = p1[st==1]
#                 good_old = p0[st==1]

#                 # draw the tracks

#                 img = 
#                 for i,(new,old) in enumerate(zip(good_new,good_old)):
#                     a,b = new.ravel()
#                     c,d = old.ravel()
#                     img = cv2.line(cart_img, (a,b),(c,d), (0,255,0), 2)
#                     img = cv2.circle(img,(a,b),5,(0,0,255),-1)
#                 cv2.imshow('Optical Flow',img)
#                 if cv2.waitKey(1) & 0xFF == ord("q"):
#                     break

#         prev_frame = cart_img.copy()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
