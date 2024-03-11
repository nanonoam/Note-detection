import cv2
import numpy as np
from typing import List, Tuple

global img
HSV_LOW_BOUND = np.array([0, 0, 0])
HSV_HIGH_BOUND = np.array([255, 255, 255])
NEXT = 0
PREVIOUS = 1
FIRST_CHILD = 2
PARENT = 3
expention = 5

def find_largest_contour_and_child(contours: List[np.ndarray], hierarchy: List[np.ndarray]) -> Tuple[int, int]:
    """Find the largest contour index and his child index

    Args:
        contours (list[np.ndarray]): The countours list
        hierarchy (list[np.ndarray]): The respective heirarchy list

    Returns:
        (int, int): the indexes of the largest contour and his child
    """
    largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    child_index = hierarchy[largest_contour_index][FIRST_CHILD]
    biggest_child_contour_index = -1
    biggest_child_contour_area = 0
    while child_index != -1:
        child_contour = contours[child_index]
        child_contour_area = cv2.contourArea(child_contour)
        if child_contour_area > biggest_child_contour_area:
            biggest_child_contour_area = child_contour_area
            biggest_child_contour_index = child_index
        child_index = hierarchy[child_index][NEXT]
    return (largest_contour_index, biggest_child_contour_index)

def get_hsv_values(img, x, y):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv[y, x]
    return hsv

def expand_hsv_bounds(img, contours, hsv_low_bound, hsv_high_bound, neighborhood_size=5, tolerance=10):
    """
    Expand the HSV value bounds based on the pixels in the contours and their neighborhood.

    Args:
        img (np.ndarray): The input image.
        contours (list): A list of contours.
        hsv_low_bound (np.ndarray): The current lower HSV bound.
        hsv_high_bound (np.ndarray): The current upper HSV bound.
        neighborhood_size (int): The size of the neighborhood around each contour pixel.
        tolerance (int): The tolerance value for including neighboring pixels in the bounds.

    Returns:
        tuple: A tuple containing the updated HSV low and high bounds.
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    new_low_bound = hsv_low_bound.copy()
    new_high_bound = hsv_high_bound.copy()

    for contour in contours:
        for point in contour:
            x, y = point[0]
            for nx in range(max(0, x - neighborhood_size // 2), min(img.shape[1], x + neighborhood_size // 2 + 1)):
                for ny in range(max(0, y - neighborhood_size // 2), min(img.shape[0], y + neighborhood_size // 2 + 1)):
                    pixel_hsv = hsv_img[ny, nx]
                    if np.all(np.abs(pixel_hsv - hsv_low_bound) <= tolerance) and np.all(np.abs(pixel_hsv - hsv_high_bound) <= tolerance):
                        new_low_bound = np.minimum(new_low_bound, pixel_hsv - tolerance)
                        new_high_bound = np.maximum(new_high_bound, pixel_hsv + tolerance)

    return new_low_bound, new_high_bound

# Click event
def click_event(event, x, y, flags, params):
    # Left click
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        hsv = get_hsv_values(img, x, y)
        global HSV_LOW_BOUND, HSV_HIGH_BOUND
        HSV_LOW_BOUND = np.array([hsv[0] - expention, hsv[1] - expention, hsv[2] - expention])
        HSV_HIGH_BOUND = np.array([hsv[0] + expention, hsv[1] + expention, hsv[2] + expention])

while (1):
    img = cv2.imread('colorperesite.PNG')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create a mask based on the specified HSV color range
    mask = cv2.inRange(hsv, HSV_LOW_BOUND, HSV_HIGH_BOUND)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Expand the HSV bounds based on the contours and their neighborhoods
    HSV_LOW_BOUND, HSV_HIGH_BOUND = expand_hsv_bounds(img, contours, HSV_LOW_BOUND, HSV_HIGH_BOUND)

    # Draw all contours on the original image
    cv2.drawContours(img, contours, -1, (255, 0, 255), 3)

    # Process only if there are contours detected
    if len(contours) != 0:
        hierarchy = hierarchy[0]
        # Find the largest contour and child contour within the largest contour
        largest_contour_index, biggest_child_contour_index = find_largest_contour_and_child(contours, hierarchy)
        cv2.drawContours(img, contours, largest_contour_index, (255, 0, 0), 5)
        # Draw the largest child contour
        if biggest_child_contour_index != -1:
            biggest_child_contour = contours[biggest_child_contour_index]
            cv2.drawContours(img, [biggest_child_contour], 0, (0, 255, 0), 2)

    cv2.imshow("img", img)
    cv2.imshow("hsv", hsv)
    cv2.imshow("mask", mask)
    cv2.setMouseCallback('img', click_event)
    cv2.waitKey(100)

    # Check for key press to exit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()