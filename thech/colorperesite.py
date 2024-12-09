import cv2
import numpy as np
import math
from typing import List, Tuple

global img
HSV_LOW_BOUND = np.array([0, 0, 0])
HSV_HIGH_BOUND = np.array([255, 255, 255])

def get_hsv_values(img, x, y):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv[y, x]
    return hsv

def expand_hsv_bounds(img, contour, hsv_low_bound, hsv_high_bound, neighborhood_size=10, tolerance=20, mouse_points=None):
    """
    Expand the HSV value bounds based on the pixels in the contours, their neighborhood, and additional mouse points.

    Args:
        img (np.ndarray): The input image.
        contour (np.ndarray): A contour representing the region of interest.
        hsv_low_bound (np.ndarray): The current lower HSV bound.
        hsv_high_bound (np.ndarray): The current upper HSV bound.
        neighborhood_size (int): The size of the neighborhood around each contour pixel.
        tolerance (int): The tolerance value for including neighboring pixels in the bounds.
        mouse_points (list): A list of (x, y) coordinates from mouse clicks.

    Returns:
        tuple: A tuple containing the updated HSV low and high bounds.
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    updated_low_bound = hsv_low_bound.copy()
    updated_high_bound = hsv_high_bound.copy()

    for pixel in contour:
        x, y = pixel[0]
        for neighbor_x in range(max(0, x - neighborhood_size // 2), min(img.shape[1], x + neighborhood_size // 2 + 1)):
            for neighbor_y in range(max(0, y - neighborhood_size // 2), min(img.shape[0], y + neighborhood_size // 2 + 1)):
                neighbor_hsv = hsv_img[neighbor_y, neighbor_x]
                if np.all(np.abs(neighbor_hsv - hsv_low_bound) <= tolerance) or np.all(np.abs(neighbor_hsv - hsv_high_bound) <= tolerance):
                    updated_low_bound = np.minimum(updated_low_bound, neighbor_hsv - tolerance)
                    updated_high_bound = np.maximum(updated_high_bound, neighbor_hsv + tolerance)


    return updated_low_bound, updated_high_bound

# Click event
def click_event(event, x, y, flags, params):
    # Left click
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        hsv = get_hsv_values(img, x, y)
        global HSV_LOW_BOUND, HSV_HIGH_BOUND
        HSV_LOW_BOUND = np.array([hsv[0], hsv[1], hsv[2]])
        HSV_HIGH_BOUND = np.array([hsv[0], hsv[1], hsv[2]])

while (1):
    img = cv2.imread('colorperesite.PNG')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create a mask based on the specified HSV color range
    mask = cv2.inRange(hsv, HSV_LOW_BOUND, HSV_HIGH_BOUND)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    

    # Draw all contours on the original image
    #cv2.drawContours(img, contours, -1, (255, 0, 255), 3)

    if len(contours) != 0:

        HSV_LOW_BOUND, HSV_HIGH_BOUND = expand_hsv_bounds(img, contours[0], HSV_LOW_BOUND, HSV_HIGH_BOUND)


    cv2.imshow("img", img)
    cv2.imshow("mask", mask)

    cv2.setMouseCallback('img', click_event)

    # Check for key press to exit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
print (HSV_LOW_BOUND)
print(HSV_HIGH_BOUND)
cv2.destroyAllWindows()