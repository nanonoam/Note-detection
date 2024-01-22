import numpy as np
import cv2

# Define the number of corners in the calibration pattern
pattern_size = (7, 7)  # Change this according to your calibration pattern

# Create arrays to store object points and image points from all calibration images
obj_points = []  # 3D points in real-world coordinatesqq
img_points = []  # 2D points in image plane coordinates

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Open video capture
cap = cv2.VideoCapture(1)

while True:
    # Read frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_frame, pattern_size, None)

    # If corners are found, add object points and image points
    if ret:
        if cv2.waitKey(1) & 0xFF == ord('s'):
        
            obj_points.append(objp)
            img_points.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
        cv2.imshow("Calibration", frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Perform camera calibration
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_frame.shape[::-1], None, None)

# Print the camera matrix and distortion coefficients
print("Camera Matrix:")
print(camera_matrix)
print("Distortion Coefficients:")
print(distortion_coeffs)

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()