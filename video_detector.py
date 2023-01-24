# PRESS Q TO QUIT WHILE RUNNING
# DETECTS FACES FROM WEBCAM FEED

import cv2

# load pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# capture webcam video
webcam = cv2.VideoCapture(0)

# iterate forever over frames
while True:
    # read current frame
    successful_frame_read, frame = webcam.read()
    
    # turn frame to greyscale
    greyscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(greyscaled_frame)
    
    # draw rectangels around faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # show frame
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # stop if Q key is pressed
    # 81 and 113 are ASCII for Q/q
    if key == 81 or key == 113:
        break

# release webcam
webcam.release()