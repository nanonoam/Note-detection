import cv2
import numpy as np
import math
# global variables go here:
field_of_view = (63.3,49.7)
#for trigo
cam_hight = 47.9 #in cm
cam_angle = 60.5
# Define lower and upper bounds for HSV color range
hsv_low = np.array([0, 95, 119], np.uint8)
hsv_high = np.array([179, 255, 255], np.uint8)
# Define hierarchy indices
NEXT = 0
PREVIOUS = 1                
FIRST_CHILD = 2
PARENT = 3
# Define kernels for smuthing and seperating donuts
smuthing_kernel = np.ones((5,5),np.float32)/25
# erode_kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
erode_kernel = np.ones((5, 5), np.uint8)

def find_largest_contour_and_child(contours, hierarchy):
    
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


#find x and y angles of note
def calculat_angle(fov,center ,frame, cam_angle):
    Angle = (fov[0]/frame.shape[1])*((frame.shape[1]/2)-center[0])
    Angle_y = (fov[1]/frame.shape[0])*((frame.shape[0]/2)-center[1]) + cam_angle
    return Angle, Angle_y
def calculat_distence(Angle_y, cam_hight):
    dist = cam_hight*np.tan(np.radians(Angle_y))
    return dist


# runPipeline() is called every frame by Limelight's backend.
def runPipeline(image, llrobot):
    dist = 0
    Angle = 0
    #blur the imagee to smooth it
    image = cv2.filter2D(image,-1,smuthing_kernel)
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Create a mask based on the specified HSV color range
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
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
                    Angle, Angle_y = calculat_angle(field_of_view, inner_center ,image, cam_angle)
                    dist = calculat_distence(Angle_y,cam_hight)
                    print(dist)

        else:
            print("There is no child contour :(")

    llpython = [dist,Angle,0,0,0,0,0,0]
       
    return contours, image, llpython