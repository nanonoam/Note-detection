import cv2
import numpy as np
import math
from typing import List, Tuple

HLS_LOW_BOUND = np.array([15, 130, 235], np.uint8)
HLS_HIGH_BOUND = np.array([40, 225, 255], np.uint8)

# Define kernels for smoothing and separating donuts
SMOOTHING_KERNEL = np.ones((5,5),np.float32)/25
ERODE_KERNEL = np.ones((7, 7), np.uint8)

MARKER2_POS = (102, 170)
MARKER1_POS = (99, 182)

# Reference marker expected HLS values
MARKER1_COLOR = (50, 205, 50) # #32cd32 Bright green
MARKER2_COLOR = (25, 255, 255) # #ffff19 Vivid Yellow

def get_optimal_lightness(image, ref_pos_yellow, ref_pos_green, ref_color_yellow, ref_color_green):
    # Get reference color RGB values
    ref_r_yellow, ref_g_yellow, ref_b_yellow = cv2.mean(ref_color_yellow)[:3]
    ref_r_green, ref_g_green, ref_b_green = cv2.mean(ref_color_green)[:3]

    # Convert reference colors to HLS
    ref_hls_yellow = cv2.cvtColor(np.uint8([[[ref_b_yellow, ref_g_yellow, ref_r_yellow]]]), cv2.COLOR_BGR2HLS)[0, 0]
    ref_hls_green = cv2.cvtColor(np.uint8([[[ref_b_green, ref_g_green, ref_r_green]]]), cv2.COLOR_BGR2HLS)[0, 0]

    # Sample regions around the reference positions
    x_yellow, y_yellow = ref_pos_yellow
    x1_yellow = max(x_yellow - 1, 0)
    y1_yellow = max(y_yellow - 1, 0)
    x2_yellow = min(x_yellow + 1, image.shape[1] - 1)
    y2_yellow = min(y_yellow + 1, image.shape[0] - 1)

    x_green, y_green = ref_pos_green
    x1_green = max(x_green - 1, 0)
    y1_green = max(y_green - 1, 0)
    x2_green = min(x_green + 1, image.shape[1] - 1)
    y2_green = min(y_green + 1, image.shape[0] - 1)

    image_patch_yellow = image[y1_yellow:y2_yellow, x1_yellow:x2_yellow]
    image_patch_green = image[y1_green:y2_green, x1_green:x2_green]

    # Convert image patches to HLS
    hls_image_patch_yellow = cv2.cvtColor(image_patch_yellow, cv2.COLOR_BGR2HLS)
    hls_image_patch_green = cv2.cvtColor(image_patch_green, cv2.COLOR_BGR2HLS)

    # Calculate the mean lightness values for the image patches
    mean_lightness_yellow = np.mean(hls_image_patch_yellow[:, :, 1])
    mean_lightness_green = np.mean(hls_image_patch_green[:, :, 1])

    # Calculate the optimal lightness based on the reference and image patch lightness values
    optimal_lightness = (ref_hls_yellow[1] + ref_hls_green[1] - mean_lightness_yellow - mean_lightness_green) / 2

    return int(optimal_lightness)

# runPipeline() is called every frame by Limelight's backend.
def runPipeline(image, llrobot):
    optimal_lightness = get_optimal_lightness(image, MARKER1_POS, MARKER2_POS, MARKER1_COLOR, MARKER2_COLOR)
    print(f"Optimal lightness: {optimal_lightness}")

    # Convert image to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # Create a mask based on the specified HLS color range
    mask = cv2.inRange(hls, HLS_LOW_BOUND, HLS_HIGH_BOUND)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours on the original image
    cv2.drawContours(image, contours, -1, (255, 0, 255), 3)

    cv2.circle(image, MARKER1_POS, 5, MARKER1_COLOR, -1)
    cv2.circle(image, MARKER2_POS, 5, MARKER2_COLOR, -1)

    cv2.imshow("image", image)

    return contours, image

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    runPipeline(img, 5)

    # Check for key press to exit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Release the video capture and destroy all windows
cv2.destroyAllWindows()