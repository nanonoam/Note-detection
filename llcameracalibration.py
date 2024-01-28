import cv2
import numpy as np


# Define the number of corners in the calibration pattern
pattern_size = (7, 7)  # Change this according to your calibration pattern

# Create arrays to store object points and image points from all calibration images
obj_points = []  # 3D points in real-world coordinatesqq
img_points = []  # 2D points in image plane coordinates

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

def runPipeline(image, llrobot):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 0, 255), 3)
    llpython = [0,0,0,0,0,0,0,0]

    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray_frame, pattern_size, None)

    # If corners are found, add object points and image points
    if ret:
        if cv2.waitKey(1) & 0xFF == ord('s'):
        
            obj_points.append(objp)
            img_points.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(image, pattern_size, corners, ret)
        cv2.imshow("Calibration", image)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_frame.shape[::-1], None, None)

        # Print the camera matrix and distortion coefficients
        print("Camera Matrix:")
        print(camera_matrix)
        print("Distortion Coefficients:")
        print(distortion_coeffs)
    
    return contours, image, llpython