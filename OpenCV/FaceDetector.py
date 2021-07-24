import cv2
from random import randrange

#load some pretrained data on frontal faces from opencv(Haar Cascade aligortihm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#load data on smiles
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')



#Choose an image to detact face from
#img = cv2.imread('images/Test.jpeg')
#img = cv2.imread('images/Stock_Family.jpeg')

## Capture video from webcam
webcam = cv2.VideoCapture(0)

#iterate all frames webcam captured
while True:
    #Read current frame from web cam
    successful_frame_read, frame = webcam.read()

    #If there is an error reading current frame form web cam
    if not successful_frame_read:
        break
    #convert to greyscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #(x, y, w, h) = face_coordinates[0]
    for (x,y,w,h) in face_coordinates:
        ##Draw rectangles to frame the faces
        cv2.rectangle(frame, (x, y),  (x + w, y + h),(0, 255, 0), 5)


        #Create a sub image to detect smile in by slicing the array
        detected_face = frame[y:y+h , x:x+w]
        #Grayscale the subimage
        grayscaled_detected_face = cv2.cvtColor(detected_face,cv2.COLOR_BGR2GRAY)
        
        #Detect the smile in the greyscale face
        smile_coordinates = smile_detector.detectMultiScale(grayscaled_detected_face, scaleFactor=1.7, minNeighbors=20)
        """"
        #Draw Rectangle to frame the smile in the detected face
        for(x_s, y_s, w_s, h_s) in smile_coordinates:
            cv2.rectangle(detected_face, (x_s, y_s),  (x_s + w_s, y_s + h_s),(0, 0, 255), 5)
        """
        #Label detected face with "Smiling" instead of drawing the rectangle to identify the smile
        if len(smile_coordinates) > 0:
            cv2.putText(frame, "Smiling", (x, y+h+40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255,255,255), thickness=2)



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
cv2.destroyAllWindows()

print("FaceDectector.py Completed")