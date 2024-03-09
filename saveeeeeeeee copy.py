import cv2
import numpy as np
import math
from typing import List, Tuple

# global variables go here: 
FIELD_OF_VIEW = (63.3,49.7) # (x degrees, y degrees)
#for trigo
CAM_HEIGHT = 61.5 #in cm
CAM_PITCH_ANGLE = 48 #in degrees
CAM_YAW_ANGLE=24
X_OFFSET = 30 #in cm
Y_OFFSET = 31.3 #in cm
# Define lower and upper bounds for HSV color range
HSV_LOW_BOUND = np.array([0, 100, 100], np.uint8)
HSV_HIGH_BOUND = np.array([30, 255, 255], np.uint8)
# Define hierarchy indices
NEXT = 0
PREVIOUS = 1                
FIRST_CHILD = 2
PARENT = 3
# Define kernels for smuthing and seperating donuts
SMOOTHING_KERNEL = np.ones((5,5),np.float32)/25
# ERODE_KERNEL = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
ERODE_KERNEL = np.ones((7, 7), np.uint8)

MARKER1_POS = (69, 190)
MARKER2_POS = (73, 178)  

# Reference marker expected HLS values
MARKER1_COLOR = (50, 205, 50) # #32cd32 Bright green
MARKER2_COLOR = (25, 255, 255) # #ffff19 Vivid Yellow


#find x and y angles of note

def calculate_angle(fov: Tuple[float, float], center: Tuple[int, int], frame: np.ndarray, camera_pitch_angle: float, cam_yaw):
    """Calculate the relative angle from the camera to the note in both axes
    -
    
    Args:
        - `fov (tuple[float, float]):` The field of view of the camera (angle_x, angle_y).
        - `center (tuple[int, int]):` the coordinates of the center of the note in the frame (x, y).
        - `frame (np.ndarray):` the frame which the note is in.
        - `camera_pitch_angle (float):` the pitch angle relative to the face of the camera.

    Returns:
        - `(angle_x, angle_y)`: the angles in both axes from the camera to the note
    """
    angle_x = (fov[0] / frame.shape[1]) * ((frame.shape[1] / 2) - center[0]) - cam_yaw       

    angle_y = (fov[1] / frame.shape[0]) * ((frame.shape[0] / 2) - center[1]) + camera_pitch_angle
    return angle_x, angle_y

def calculate_distance(angle_y: float, cam_height: float):
    """Calculates the distance from the camera to the note in the X axis
    -
    Args:
        - `angle_y (float):` the y axis angle from the note to the camera
        - `cam_height (float):` the height of the camera relative to the ground 

    Returns:
        `The distance in the X axis from the camera to the note.`
    """ 
    return cam_height*np.tan(np.radians(angle_y))

def convert_to_mid_of_robot(llpython: list, x_offset: int, y_offset: int):
    """Convert the distance and angle from the camera into the distance from the robot.

    Args:
        llpython (list): the output limelight array
        x_offset (int): the offset of the camera from the center of the robot
        y_offset (int): the offset of the camera from the center of the robot

    Returns:
        `llpython`: The output array for the limelight 
    """
    
    distance = llpython[0]
    angle = llpython[1]
    angle_rad = math.radians(angle)
    y = distance * math.sin(angle_rad) + y_offset
    x = distance * math.cos(angle_rad) + x_offset
    angle = math.degrees(math.atan(y/x))
    llpython = [math.hypot(x,y), angle]
    return llpython


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
    return (largest_contour_index ,biggest_child_contour_index)


def get_exposure_increase(image, ref_pos, ref_color):

    # Get reference color RGB 
    ref_r, ref_g, ref_b = cv2.mean(ref_color)[:3]
    
    # Sample a 3x3 region around the pos 
    x, y = ref_pos
    x1 = max(x-1,0)
    y1 = max(y-1,0)
    x2 = min(x+1, image.shape[1]-1)
    y2 = min(y+1, image.shape[0]-1)
    
    image_patch = image[y1:y2, x1:x2]
    b, g, r = cv2.mean(image_patch)[:3]
    
    # Calculate luminances
    ref_lum = 0.299*ref_r + 0.587*ref_g + 0.114*ref_b
    img_lum = 0.299*r + 0.587*g + 0.114*b

    # Check for divide by 0
    if img_lum == 0:
        return 0
    
    ratio = ref_lum / img_lum
    
    # Keep ratio within limits
    ratio = max(ratio, 0.01)
    ratio = min(ratio, 100)
    
    # Log scale
    log_ratio = np.log2(ratio)
    
    # Round to nearest 10 ms
    ms = round(log_ratio * 100 / 10) * 10
        
    return ms


# runPipeline() is called every frame by Limelight's backend.
def runPipeline(image, llrobot):


    exposure_increase = (get_exposure_increase(image, MARKER1_POS, MARKER1_COLOR) + get_exposure_increase(image, MARKER2_POS, MARKER2_COLOR))/2
    
    dist = 0
    Angle = 0
    
    #blur the imagee to smooth it
    image = cv2.filter2D(image,-1,SMOOTHING_KERNEL)
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Create a mask based on the specified HSV color range
    mask = cv2.inRange(hsv, HSV_LOW_BOUND, HSV_HIGH_BOUND)

    mask = cv2.erode(mask,ERODE_KERNEL)
    mask = cv2.dilate(mask,ERODE_KERNEL)
    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw all contours on the original image
    cv2.drawContours(image, contours, -1, (255, 0, 255), 3)
    # Process only if there are contours detected
    if len(contours) != 0:
        hierarchy = hierarchy[0]
        # Find the largest contour and child contour within the largest contour
        largest_contour_index, biggest_child_contour_index = find_largest_contour_and_child(contours, hierarchy) 
        cv2.drawContours(image, contours, largest_contour_index, (255, 0, 0), 5)
        # Draw the largest child contour
        if biggest_child_contour_index != -1:
            biggest_child_contour = contours[biggest_child_contour_index]
            cv2.drawContours(image, [biggest_child_contour], 0, (0, 255, 0), 2)

            outer_contour = contours[largest_contour_index]
            inner_contour = contours[biggest_child_contour_index]

            # Check if both outer and inner contours have areas greater than 3600
            if (cv2.contourArea(outer_contour) > 100) and (cv2.contourArea(inner_contour) > 100):
                x, y, w, h = cv2.boundingRect(outer_contour)
                outer_aspect_ratio = float(w) / h
                outer_center = (int(x+(w/2)), int(y+(h/2)))
                cv2.circle(image, outer_center, 10, (255,255,255),-3)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 5)

                x1, y1, w1, h1 = cv2.boundingRect(inner_contour)
                inner_aspect_ratio = float(w1) / h1
                inner_center = (int(x1+(w1/2)), int(y1+(h1/2)))
                cv2.circle(image, inner_center, 5, (0,0,0),-3)
                cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 5)
            
                # Check if the aspect ratios and distance between centers meet the criteria
                if (abs(outer_aspect_ratio - inner_aspect_ratio) < 10) and (math.dist(outer_center, inner_center) < 20):
                    image = cv2.putText(image, 'probably donut?☺☻♥', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
                    Angle, Angle_y = calculate_angle(FIELD_OF_VIEW, inner_center ,image, CAM_PITCH_ANGLE, CAM_YAW_ANGLE)
                    dist = calculate_distance(Angle_y,CAM_HEIGHT)
                    
                else:
                    cv2.putText(image, 'coected note pls bump to seperate', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        else:
            print("There is no child contour :(")

    llpython = [dist,Angle, exposure_increase]
    if (llpython[0] > 0):
        llpython = convert_to_mid_of_robot(llpython, X_OFFSET, Y_OFFSET)

    cv2.circle(image, MARKER1_POS, 5, MARKER1_COLOR,-1)
    cv2.circle(image, MARKER2_POS, 5, MARKER2_COLOR,-1)

    cv2.imshow("image",image)

    print(llpython)

    return contours, image, llpython

cap = cv2.VideoCapture(0)

while(1):
    
    ret, img = cap.read()

    runPipeline(img, 5)

    # Check for key press to exit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Release the video capture and destroy all windows
cv2.destroyAllWindows()