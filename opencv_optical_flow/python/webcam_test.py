import cv2
import numpy as np

try:
    # Open the webcam
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


# Read the first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Parameters for optical flow calculation
pyr_scale = 0.5  # Image pyramid or scale factor
levels = 3  # Number of pyramid levels
winsize = 15  # Window size for flow calculation
iterations = 3  # Number of iterations at each pyramid level
poly_n = 5  # Size of the pixel neighborhood
poly_sigma = 1.1  # Standard deviation of the Gaussian used for smoothing derivatives
flags = 0  # Additional flags (e.g., cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)

while True:
    # Read the current frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=gray,
        flow=None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags,
    )
    print(np.min(prev_gray), np.max(prev_gray))
    print(np.min(gray), np.max(gray))

    # Overlay the optical flow arrows onto the original frame
    overlay = frame.copy()
    step = 36  # Step size for drawing arrows
    for y in range(0, overlay.shape[0], step):
        for x in range(0, overlay.shape[1], step):
            dx = int(flow[y, x, 0])
            dy = int(flow[y, x, 1])
            cv2.arrowedLine(overlay, (x, y), (x + dx, y + dy), (0, 255, 0), 1)

    # Show the resulting frame
    cv2.imshow("Optical Flow", overlay)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Update the previous frame and previous gray image
    prev_frame = frame.copy()
    prev_gray = gray.copy()

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
