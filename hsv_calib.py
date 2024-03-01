# importing the module 
import cv2 

# function to display the coordinates of 
# of the points clicked on the image 
def click_event(event, x, y, flags, params): 

	# checking for left mouse clicks 
	if event == cv2.EVENT_LBUTTONDOWN: 

		# displaying the coordinates 
		# on the Shell 
		print(x, ' ', y) 

		# displaying the coordinates 
		# on the image window 
		font = cv2.FONT_HERSHEY_SIMPLEX 
		cv2.putText(img, str(x) + ',' +
					str(y), (x,y), font, 
					1, (255, 0, 0), 2) 
		cv2.imshow('image', img) 

	# checking for right mouse clicks	 
	if event==cv2.EVENT_RBUTTONDOWN: 

		# displaying the coordinates 
		# on the Shell 
		print(x, ' ', y) 

		# displaying the coordinates 
		# on the image window 
		font = cv2.FONT_HERSHEY_SIMPLEX 
		b = img[y, x, 0] 
		g = img[y, x, 1] 
		r = img[y, x, 2] 
		cv2.putText(img, str(b) + ',' +
					str(g) + ',' + str(r), 
					(x,y), font, 1, 
					(255, 255, 0), 2) 
		cv2.imshow('image', img) 

# driver function 
cap = cv2.VideoCapture(1)

while(1):
    
    ret, img = cap.read()

	# displaying the image 
    cv2.imshow('image', img) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# setting mouse handler for the image 
	# and calling the click_event() function 
    cv2.setMouseCallback('image', click_event) 

	# wait for a key to be pressed to exit 
    cv2.waitKey(0) 

	# close the window 
    cv2.destroyAllWindows() 
# the note has 2 hsv color ranges (0,70,250 - 5,80,255) and (170,90,250) - (179,110,255)