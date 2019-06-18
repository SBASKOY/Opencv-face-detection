
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:00:20 2019

@author: Salim
"""
import cv2
import os
from PIL import Image
import numpy as np
import pickle
haar_file = 'haarcascade_frontalface_default.xml'
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
images_dir=os.path.join(BASE_DIR,"İmages")
face_cas=cv2.CascadeClassifier(haar_file)

recognizer=cv2.face.LBPHFaceRecognizer_create()

x_labels=[]
x_train=[]
current_id=0
label_id={}
        
for root,dirs,files in os.walk(images_dir):
    for file in files:
        if file.endswith("png")or file.endswith("jpeg") or file.endswith("jpg") or file.endswith("JPG"):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            if not label in label_id:
                label_id[label]=current_id
                current_id+=1
            id_=label_id[label]
            
            pil_image=Image.open(path).convert("L")
            image_array=np.array(pil_image,"uint8")
            #print(image_array)
            faces = face_cas.detectMultiScale(image_array, 1.3, 4)
            for (x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                x_labels.append(id_)
                
with open ("label.pickle","wb") as f:
    pickle.dump(label_id,f)
recognizer.train(x_train,np.array(x_labels))
recognizer.save("trainner.yml")
print("Bitti")