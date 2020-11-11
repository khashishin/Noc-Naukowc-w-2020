import cv2
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread(sys.argv[1])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
blue = (255,0,0)
green = (0,255,0)

while(True):
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),blue,2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),green,2)

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        	cv2.destroyAllWindows()