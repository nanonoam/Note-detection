import cv2
import numpy as np
from networktables import NetworkTables
import cscore

# Initialize NetworkTables and get the instance
NetworkTables.initialize(server='10.56.35.2')  # Replace with your team's IP
sd = NetworkTables.getTable('SmartDashboard')

def process_image(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of orange color in HSV
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([15, 255, 255])

    # Create a mask for orange color
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Send contours to NetworkTables
    send_contours_to_networktables(contours)

    # Optional: Draw contours on the original image for visualization
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    return image

def send_contours_to_networktables(contours):
    # Convert contours to a network-friendly format
    contours_data = [[(point[0][0], point[0][1]) for point in contour] for contour in contours]

    # Send contours to NetworkTables
    sd.putValue('OrangeContours', contours_data)

# Get the camera instance
camera = cscore.CameraServer.getInstance().startAutomaticCapture()
camera.setResolution(640, 480)  # Set the desired resolution

while True:
    # Get the frame from the camera
    time, image = camera.captureFrame()

    # Process the image
    processed_image = process_image(image)

    # Display the processed image (optional)
    cv2.imshow('Processed Image', processed_image)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()