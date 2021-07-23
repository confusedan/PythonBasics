import cv2
from random import randrange

#load some pretrained data on frontal faces from opencv(Haar Cascade aligortihm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detact face from
#img = cv2.imread('images/Test.jpeg')
#img = cv2.imread('images/Stock_Family.jpeg')

## Capture video from webcam
webcam = cv2.VideoCapture(0)

#iterate all frames webcam captured
while True:
    #Read current frame
    successful_frame_read, frame = webcam.read()
    #convert to greyscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #(x, y, w, h) = face_coordinates[0]
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y),  (x + w, y + h),(randrange(255), randrange(255), randrange(255)), 5)


    cv2.imshow('WebCam', frame)
    key = cv2.waitKey(1)

    ##Stop if Q or q is pressed
    if key==81 or key==113:
        break
"""
#detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#(x, y, w, h) = face_coordinates[0]
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x, y),  (x + w, y + h),(randrange(255), randrange(255), randrange(255)), 5)
#show img
cv2.imshow('Face Detector', img)

cv2.waitKey()
"""
##Release VideoCapture Object
webcam.release()
print("FaceDectector.py Completed")