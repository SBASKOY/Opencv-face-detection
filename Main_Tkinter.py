# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:54:49 2019

@author: Salim
"""

import tkinter as tk 
from tkinter.filedialog import askopenfilename
from PIL import Image
from PIL import ImageTk
import cv2



resimler=[]


ekran=tk.Tk()
ekran.title("Görüntü İşleme")
ekran.geometry("800x500")
canvas=tk.Canvas(ekran,bg="yellow")
canvas.place(x=0,y=10,relwidth=1,relheight=1)




resimekle=tk.Button(canvas,text="Resim Yükle",command=lambda:resimaç(canvas))
resimekle.place(x=10,y=400,width=100,height=50)
bul=tk.Button(canvas,text="Yüzleri Bul",command=lambda:bul(canvas))
bul.place(x=110,y=400,width=100,height=50)
def resimaç(canvas):
    dizin=askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    resim=cv2.imread(dizin)
    resimler.append(resim)
    resim=cv2.resize(resim,(300,300))
    resim = Image.fromarray(resim)
    resim = ImageTk.PhotoImage(resim)
    
    #cv2.imshow("resim",resim)
    label1=tk.Label(canvas,image=resim)
    label1.image=resim
    label1.place(x=50,y=10,width=300,height=300)
def bul(canvas):
    import pickle
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainner.yml")
    haar_file = 'haarcascade_frontalface_default.xml' 
    face_cascade = cv2.CascadeClassifier(haar_file) 
    labels={"isim=":1}
        
    with open ("label.pickle","rb") as f:
        f_labels=pickle.load(f)
        labels={v:k for k,v in f_labels.items()}
        im=resimler[0]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
        for (x, y, w, h) in faces:      
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            face = gray[y:y + h, x:x + w] 
            face_resize = cv2.resize(face, (130, 130))
            id_,conf=recognizer.predict(face)
            if conf>=45:
                font=cv2.FONT_HERSHEY_SIMPLEX
                name=labels[id_]
                color=(255,255,255)
                stroke=1
                cv2.putText(im,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
    
    cv2.imshow("rsim",im)
    im=cv2.resize(im,(300,300))
    resim = Image.fromarray(im)
    resim = ImageTk.PhotoImage(resim)
    
    #cv2.imshow("resim",resim)
    label2=tk.Label(canvas,image=resim)
    label2.image=resim
    label2.place(x=400,y=10,width=300,height=300)
    cv2.imshow("face",face_resize)
    
    
    
ekran.mainloop()
