# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:48:48 2019

@author: Salim
"""


import cv2 
import numpy as np
import pickle


recognizer=cv2.face.LBPHFaceRecognizer_create()

recognizer.read("trainner.yml")

haar_file = 'haarcascade_frontalface_default.xml'   
(width, height) = (130, 100)     
      
face_cascade = cv2.CascadeClassifier(haar_file) 
 
      
    # The program loops until it has 30 images of the face. 
labels={"isim=":1}
        
with open ("label.pickle","rb") as f:
    f_labels=pickle.load(f)
    labels={v:k for k,v in f_labels.items()}
    
webcam = cv2.VideoCapture(0)    
while (webcam.isOpened()):  
    (_, im) = webcam.read() 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
    for (x, y, w, h) in faces: 
        
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        face = gray[y:y + h, x:x + w] 
        face_resize = cv2.resize(face, (width, height))
        id_,conf=recognizer.predict(face)
        if conf>=45:
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=1
            cv2.putText(im,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
            
    cv2.imshow('OpenCV', im)
    
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break
webcam.release()
cv2.destroyAllWindows()