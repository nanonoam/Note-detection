import cv2
import numpy as np
import math
from typing import List, Tuple

HSV_LOW_BOUND = np.array([2, 188, 163])
HSV_HIGH_BOUND = np.array([33, 255, 239])



while (1):
    img = cv2.imread('colorperesite.PNG')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create a mask based on the specified HSV color range
    mask = cv2.inRange(hsv, HSV_LOW_BOUND, HSV_HIGH_BOUND)

    
    cv2.imshow("img", img)
    cv2.imshow("mask", mask)


    # Check for key press to exit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
print (HSV_LOW_BOUND)
print(HSV_HIGH_BOUND)
cv2.destroyAllWindows()