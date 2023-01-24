# PRESS ANY KEY TO QUIT WHILE RUNNING
# DETECTS FACES FROM AN IMAGE

import cv2

# load pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# choose an image to detect faces
img = cv2.imread('rdj.jpg')

# turn image to greyscale
greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

# draw rectangels around faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# show image
cv2.imshow('Face Detector', img)
cv2.waitKey()