import ntcore
import numpy as np
import cv2
from pupil_apriltags import Detector

# מגדירים פרמטרים של המצלמה אחרי קליברציה
camera_fx = 641.00105423
camera_fy = 639.87321415
camera_cx = 340.11271572
camera_cy = 81.68636793
camera_matrix = np.array([[camera_fx, 0, camera_cx],
                          [0, camera_fy, camera_cy],
                          [0, 0, 1]])

distortion_coeffs = np.array([[-0.22803942, -0.68149206, -0.01026001, 0.0038208, 1.13613218]])

#מגדירים את התג שמישתמשים בו
detector = Detector(families='tag36h11',
                    nthreads=1,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)

inst = ntcore.NetworkTableInstance.getDefault()

inst.startClient4("example client")

# connect to a roboRIO with team number TEAM
inst.setServerTeam(5635)

# starting a DS client will try to get the roboRIO address from the DS application
inst.startDSClient()

inst.setServer("host", ntcore.NetworkTableInstance.kDefaultPort4)

# Get the table within that instance that contains the data. Correct the table name.
table = inst.getTable("datable")

cap = cv2.VideoCapture(0) #פותחים מצלמה

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # מתקן תמונה
    undistorted_frame = cv2.undistort(gray, camera_matrix, distortion_coeffs)

    tags = detector.detect(undistorted_frame, estimate_tag_pose=True, camera_params=(camera_fx, camera_fy, camera_cx, camera_cy), tag_size=0.161)
    for tag in tags:
        tagid = tag.tag_id

        tag_id = table.getEntry("tag_id")
        tag_id.setDouble(tagid)
        print(tagid)