import cv2

#Load Full body data for body tracking
trained_full_body_data = cv2.CascadeClassifier('haarcascade_fullbody.xml')
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

## Capture video from webcam
webcam = cv2.VideoCapture(0)


#iterate all frames webcam captured
while True:
    #Read current frame
    successful_frame_read, frame = webcam.read()
    #convert to greyscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect bodies
    body_coordinates = trained_full_body_data.detectMultiScale(grayscaled_img)
    #detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)


    #(x, y, w, h) = body_coordinates[0]
    for (x,y,w,h) in body_coordinates:
        cv2.rectangle(frame, (x, y),  (x + w, y + h),(255, 255, 0), 5)

    #(x, y, w, h) = face_coordinates[0]
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y),  (x + w, y + h),(0, 255, 0), 5)


    cv2.imshow('WebCam', frame)
    key = cv2.waitKey(1)

    ##Stop if Q or q is pressed
    if key==81 or key==113:
        break

##Release VideoCapture Object
webcam.release()
print("Face_Body_Detector.py Completed")