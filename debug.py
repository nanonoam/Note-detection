import numpy as np
import cv2
import math
from pynput import keyboard
import threading

is_running = True
hsv_low = np.array([0, 112, 199], np.uint8)
hsv_high = np.array([100, 255, 255], np.uint8)

# Define hierarchy indices
NEXT = 0
PREVIOUS = 1
FIRST_CHILD = 2
PARENT = 3

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


# Process video frames
def detect_note(video):
    """Detect Notes from the frc challange
    -
    
    `Args:`
        - video (`string or int`): a path to a vidoe or a camare index.
    """
    
    
    cap = cv2.VideoCapture(video)
    
    while(cap.isOpened()):
        
        global is_running
        if is_running:
            ret, img = cap.read()
            # Convert image to HSV color space
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Create a mask based on the specified HSV color range
            mask = cv2.inRange(hsv, hsv_low, hsv_high)
            # Find contours in the mask
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

                    outer_contour = contours[largest_contour_index]
                    inner_contour = contours[biggest_child_contour_index]

                    # Check if both outer and inner contours have areas greater than 3600
                    if (cv2.contourArea(outer_contour) > 3600) and (cv2.contourArea(inner_contour) > 3600):
                        x, y, w, h = cv2.boundingRect(outer_contour)
                        outer_aspect_ratio = float(w) / h
                        outer_center = (int(x+(w/2)), int(y+(h/2)))
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        x, y, w, h = cv2.boundingRect(inner_contour)
                        inner_aspect_ratio = float(w) / h
                        inner_center = (int(x+(w/2)), int(y+(h/2)))
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

                        # Check if the aspect ratios and distance between centers meet the criteria
                        if (abs(outer_aspect_ratio - inner_aspect_ratio) < 0.2) and (math.dist(outer_center, inner_center) < 10):
                            image = cv2.putText(img, 'probably donut?☺☻♥', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
                else:
                    print("There is no child contour :(")

            # Display the mask and the processed image
            cv2.imshow('mask', mask)
            cv2.imshow('img', img)

            # Check for key press to exit
            k = cv2.waitKey(10) & 0xFF
            if k == 27:
                break

    # Release the video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
 

def on_press(key): #on press commend to stop the video.
    global is_running
    try:
        if key.char == 'p':
            is_running = not is_running 
            print("Background code paused" if not is_running else "Background code resumed")
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press) #set the keyboard listener.

listener_thread = threading.Thread(target=listener.start) # put the listener on a different thread.

if __name__ == "__main__": 
    listener_thread.start() #start the listener thread. 
    detect_note("Note_Video.mp4") #start the main Tracking code.

    