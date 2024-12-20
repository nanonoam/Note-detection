import cv2
import numpy as np
import math
from typing import List, Tuple

global img
HSV_LOW_BOUND = np.array([2, 188, 163])
HSV_HIGH_BOUND = np.array([33, 255, 239])
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


while (1):
    img = cv2.imread('colorperesite.PNG')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create a mask based on the specified HSV color range
    mask = cv2.inRange(hsv, HSV_LOW_BOUND, HSV_HIGH_BOUND)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    

    # Draw all contours on the original image
    #cv2.drawContours(img, contours, -1, (255, 0, 255), 3)

    if len(contours) != 0:
        hierarchy = hierarchy[0]
        # Find the largest contour and child contour within the largest contour
        largest_contour_index, biggest_child_contour_index = find_largest_contour_and_child(contours, hierarchy) 
        cv2.drawContours(img, contours, largest_contour_index, (0, 255, 0), 5)
        # Draw the largest child contour
        if biggest_child_contour_index != -1:
            biggest_child_contour = contours[biggest_child_contour_index]
            cv2.drawContours(img, [biggest_child_contour], 0, (255, 0, 0), 2)

            outer_contour = contours[largest_contour_index]
            inner_contour = contours[biggest_child_contour_index]

        else:
            cv2.putText(img, 'There is no child contour :(', (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)


    cv2.imshow("img", img)
    cv2.imshow("mask", mask)


    # Check for key press to exit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
print (HSV_LOW_BOUND)
print(HSV_HIGH_BOUND)
cv2.destroyAllWindows()