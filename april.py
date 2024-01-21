import numpy as np
import cv2
from pupil_apriltags import Detector


# מגדירים פרמטרים של המצלמה אחרי קליברציה
camera_fx = 624.9471386
camera_fy = 635.74083359
camera_cx = 321.90189406
camera_cy = 263.79474841

field_width = 16.654 # In meters
field_height = 8.229 # In meters

atagarr = np.array([[0,1],[0,3],[0,5],[0,7],[16.654,1],[16.654,3],[16.654,5],[16.654,7]])
# path = [[0, 0], [130, 344],[198, 172],[464, 156],[463, 205]]
camera_matrix = np.array([[camera_fx, 0, camera_cx],
                          [0, camera_fy, camera_cy],
                          [0, 0, 1]])

distortion_coeffs = np.array([[-4.11082203e-01, 3.16018361e+00, 4.99088110e-03, -1.10638717e-02, -8.65431343e+00]])


#מגדירים את התג שמישתמשים בו
detector = Detector(families='tag36h11',
                    nthreads=1,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)
cap = cv2.VideoCapture(0) #פותחים מצלמה

while True:
    #צירים תמונה שחורה
    black_image = np.zeros((int(field_width*20), int(field_height*20), 3), dtype=np.uint8)
    for idx, atag in enumerate(atagarr):
        cv2.circle(black_image, (int(20*atag[1]), int(20*atag[0])), 10, (255, 255, 255), -1)
        cv2.putText(black_image, str(idx), (int(20 * atag[1]), int(20 * atag[0])+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)


    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # מתקן תמונה
    undistorted_frame = cv2.undistort(gray, camera_matrix, distortion_coeffs)

    tags = detector.detect(undistorted_frame, estimate_tag_pose=True, camera_params=(camera_fx, camera_fy, camera_cx, camera_cy), tag_size=0.055)
    for tag in tags:
        
        distance = tag.pose_t[2]  # עומק לטאג
        sidedist = tag.pose_t[0] #ימין שמאל לטאג
        #ממירים לסנטימטרים
        cmsidedist = sidedist*100
        cmdist = distance*100
        #מראים את הנקודה
        cv2.circle(black_image, (250 + int(cmsidedist),int(cmdist)), 5, (255,0,0), 2)
        # print(f"tag id: {tag.tag_id} Distance to tag: {int(distance)} mm")

        # איפה הטאג
        for idx in range(len(tag.corners)):
            cv2.line(frame, tuple(tag.corners[idx - 1, :].astype(int)),
                        tuple(tag.corners[idx, :].astype(int)), (0, 255, 0), 2)
    cv2.imshow('Black Image', black_image)
    cv2.imshow('Tag Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
