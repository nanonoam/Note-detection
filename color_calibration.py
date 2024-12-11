import cv2
import numpy as np

# Trackbar callback function to update HLS value
def callback(x):
    global H_low, H_high, L_low, L_high, S_low, S_high
    # Assign trackbar position value to H, L, S High and low variable
    H_low = cv2.getTrackbarPos('low H', 'controls')
    H_high = cv2.getTrackbarPos('high H', 'controls')
    L_low = cv2.getTrackbarPos('low L', 'controls')
    L_high = cv2.getTrackbarPos('high L', 'controls')
    S_low = cv2.getTrackbarPos('low S', 'controls')
    S_high = cv2.getTrackbarPos('high S', 'controls')

# Create a separate window named 'controls' for trackbar
cv2.namedWindow('controls', 2)
cv2.resizeWindow("controls", 550, 10)

# Global variable
H_low = 0
H_high = 179
L_low = 0
L_high = 255
S_low = 0
S_high = 255

# Create trackbars for high, low H, L, S
cv2.createTrackbar('low H', 'controls', 0, 179, callback)
cv2.createTrackbar('high H', 'controls', 179, 179, callback)

cv2.createTrackbar('low L', 'controls', 0, 255, callback)
cv2.createTrackbar('high L', 'controls', 255, 255, callback)

cv2.createTrackbar('low S', 'controls', 0, 255, callback)
cv2.createTrackbar('high S', 'controls', 255, 255, callback)

# Initialize webcam video capture
webcam_video = cv2.VideoCapture(0)

while True:
    # Read source image from webcam
    success, img = webcam_video.read()
    if not success:
        print("Failed to capture image from webcam.")
        break

    # Convert source image to HLS color mode
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hls_low = np.array([H_low, L_low, S_low], np.uint8)
    hls_high = np.array([H_high, L_high, S_high], np.uint8)

    # Making mask for hls range
    mask = cv2.inRange(hls, hls_low, hls_high)
    # Masking HLS value selected color becomes black
    res = cv2.bitwise_and(img, img, mask=mask)

    # Show image
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('hls', hls)

    # Wait for the user to press escape and break the while loop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Release the webcam and destroy all windows
webcam_video.release()
cv2.destroyAllWindows()
