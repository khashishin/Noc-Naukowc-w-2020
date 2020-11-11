import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT

blue = (255,0,0)
green = (0,255,0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Our operations on the frame come here
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # print(len(faces))
    # Display the resulting frame
    for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),blue,2)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = frame[y:y+h, x:x+w]
         eyes = eye_cascade.detectMultiScale(roi_gray)
         for (ex,ey,ew,eh) in eyes:
             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),green,2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()












# import cv2
# import sys
# import logging as log
# import datetime as dt
# from time import sleep

# cascPath = "\\haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)
# log.basicConfig(filename='webcam.log',level=log.INFO)

# video_capture = cv2.VideoCapture(0)
# anterior = 0

# while True:
#     if not video_capture.isOpened():
#         print('Unable to load camera.')
#         sleep(5)
#         pass

#     # Capture frame-by-frame
#     ret, frame = video_capture.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Draw a rectangle around the faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     if anterior != len(faces):
#         anterior = len(faces)
#         log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


#     # Display the resulting frame
#     cv2.imshow('Video', frame)


#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     # Display the resulting frame
#     cv2.imshow('Video', frame)

# # When everything is done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()