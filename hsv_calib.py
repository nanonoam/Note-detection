# importing OpenCV and aliasing 
import cv2

# Global img variable
global img

# Click event 
def click_event(event, x, y, flags, params):

    # Left click 
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX 

        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' + str(y), (x,y), font, 1, (255, 0, 0), 2)
        cv2.imshow('image', img)

    # Right click
    elif event == cv2.EVENT_RBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX 
        b = img[y, x, 0]
        g = img[y, x, 1] 
        r = img[y, x, 2]

        cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x,y), font, 1, (255, 255, 0), 2) 
        cv2.imshow('image', img)
           
# Video capture
cap = cv2.VideoCapture(0) 

# Loop
while True:
    ret, img = cap.read() 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)  
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(1)

# Release capture  
cap.release()
cv2.destroyAllWindows()