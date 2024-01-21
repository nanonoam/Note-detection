import cv2  
import numpy as np
import math
hsv_low = np.array([0, 80, 178], np.uint8)
hsv_high = np.array([179, 255, 255], np.uint8)

webcam_video = cv2.VideoCapture(1)

#[next, previous, first child, parent] --> hierarchy 
NEXT = 0
PREVIOUS = 1
FIRST_CHILD = 2
PARENT = 3


while(1):

    success, img = webcam_video.read()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, hsv_low, hsv_high)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img, contours, -1, (255, 0, 255), 3)


    if len(contours) != 0:
        hierarchy = hierarchy[0]
        #[next, previous, first child, parent] ()-> hierarchy 

        largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))

        cv2.drawContours(img, contours, largest_contour_index, (255, 0, 0))
        cv2.drawContours(img, contours, hierarchy[largest_contour_index][PREVIOUS], (0, 255, 255), 3)
        cv2.drawContours(img, contours, hierarchy[largest_contour_index][NEXT], (0, 0, 255), 3)

        print(hierarchy[largest_contour_index])
        if hierarchy[largest_contour_index][FIRST_CHILD] != -1:
            
            print("HERE")

            outer_contour = contours[largest_contour_index]
            inner_contour = contours[hierarchy[largest_contour_index][FIRST_CHILD]]

            if (cv2.contourArea(outer_contour) > 3600) and (cv2.contourArea(inner_contour) > 3600) :

                    x, y, w, h = cv2.boundingRect(outer_contour)
                    outer_aspect_ratio = float(w) / h
                    outer_center = (int(x+(w/2)), int(y+(h/2)))
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    x, y, w, h = cv2.boundingRect(inner_contour)
                    inner_aspect_ratio = float(w) / h
                    inner_center = (int(x+(w/2)), int(y+(h/2)))
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

                    if (abs(outer_aspect_ratio - inner_aspect_ratio) < 0.2) and (math.dist(outer_center, inner_center) < 10):
                        image = cv2.putText(img, 'probobly donut?☺☻♥', (x, y), cv2.FONT_HERSHEY_SIMPLEX ,  1, (255, 0, 255), 2, cv2.LINE_AA) 
                        X_Angle = (img.shape[0]/85)*np.abs((img.shape[0]/2) - (x+(int(w/2))))
                        #Y_Angle = (img.shape[1]/fov[1])*np.abs((img.shape[1]/2) - (y+(int(h/2))))
                        print(X_Angle)

    cv2.imshow('mask',mask)
    cv2.imshow('img',img)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()