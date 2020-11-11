# Face filters (Snapchat like) using OpenCV
# @author:- Kunal Gupta (cite as kg777)
# source: https://github.com/kunalgupta777/OpenCV-Face-Filters/blob/master/filters.py
import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np
import os
import subprocess

cascPath = "haarcascade_frontalface_default.xml"  # for face detection
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
mst = cv2.imread('moustache.png')
hat = cv2.imread('cowboy_hat.png')
dog = cv2.imread('dog_filter.png')

def put_moustache(mst,fc,x,y,w,h):    
    face_width = w
    face_height = h

    mst_width = int(face_width*0.4166666)+1
    mst_height = int(face_height*0.142857)+1
    mst = cv2.resize(mst,(mst_width,mst_height))

    for i in range(int(0.62857142857*face_height),int(0.62857142857*face_height)+mst_height):
        for j in range(int(0.29166666666*face_width),int(0.29166666666*face_width)+mst_width):
            for k in range(3):
                if mst[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k] <235:
                    fc[y+i][x+j][k] = mst[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k]
    return fc

def put_hat(hat,fc,x,y,w,h):
    
    face_width = w
    face_height = h    
    hat_width = face_width
    hat_height = int(0.6*face_height)
    
    hat = cv2.resize(hat,(hat_width,hat_height))
    
    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if hat[i][j][k]<235:
                    fc[y+i-int(0.52*face_height)][x+j][k] = hat[i][j][k]
    return fc

def put_dog_filter(dog,fc,x,y,w,h):
    face_width = w
    face_height = h
    
    width_scale = 2.45#1.5
    height_scale = 2.9#1.75

    width = int(face_width*width_scale)
    height = int(face_height*height_scale)    

    dog = cv2.resize(dog,(width,height))
    for i in range(height):
        for j in range(width):
            for k in range(3):
                if dog[i][j][k]<235:
                    fc[y+i-int(0.92*h)-1][x+j-int(0.77*w)][k] = dog[i][j][k]
    return fc
    

ch = 0
print ("Select Filter:1.) Hat 2.) Moustache 3.) Hat and Moustache 4.) Dog Filter")
ch = int(input())    
    
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40,40)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        if ch==2:
            frame = put_moustache(mst,frame,x,y,w,h)
        elif ch==1:
            frame = put_hat(hat,frame,x,y,w,h)
        elif ch==3:
            frame = put_moustache(mst,frame,x,y,w,h)
            frame = put_hat(hat,frame,x,y,w,h)
        else:
            frame = put_dog_filter(dog,frame,x,y,w-25,h+20)
            
    if anterior != len(faces):
        anterior = len(faces)
        # log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
