import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
from tkinter import *
from tkinter import filedialog
import cv2
from PIL import ImageTk, Image
from detection import AccidentDetectionModel
import numpy as np
import os
from tkinter import *
from tkinter import filedialog
import time
from plyer import notification
import customtkinter as ctk
from PIL import ImageTk, Image


# root = ctk.ThemedTk()
# root.set_theme("arc")

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX
ctk.set_appearance_mode("dark")

# Selecting color theme - blue, green, dark-blue
ctk.set_default_color_theme("blue")
app = ctk.CTk()
app.geometry("250x250")
app.title("God's Eye")

label = ctk.CTkLabel(app,text=" Select the option ")
label.pack()


def open_camera():
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if(pred == "Accident"):
            prob = (round(prob[0][0]*100, 2))

            # to beep when alert:
            # if(prob > 90):
            #     os.system("say beep")

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            return
        cv2.imshow('Video', frame)
    video.release()
    cv2.destroyAllWindows()

def select_folder():
    global location
    file_path = filedialog.askopenfilename(title="Select a video file")
    return file_path.replace('/', '\\')


def startapplication():
    path=select_folder()
    video = cv2.VideoCapture(path) # for camera use video = cv2.VideoCapture(0)
    while True: 
        ret, frame = video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if(pred == "Accident"):
            prob = (round(prob[0][0]*100, 2))
            
            # to beep when alert:
            if(prob > 85):
                notification.notify(
                title = "ALERT!",
                message = "accident detected" 
                ,  timeout=.3
                )
            # if(prob > 10):
            #     os.system("\n\n\n\n\n")

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            return
        cv2.imshow('Video', frame)  

frame = ctk.CTkFrame(master=app)
frame.pack(pady=20,padx=40,fill='both',expand=True)

button = ctk.CTkButton(master=frame,text='Video',command=start_application)
button.pack(pady=12,padx=10)
button = ctk.CTkButton(master=frame,text='CCTV',command=open_camera)
button.pack(pady=12,padx=10)
app.mainloop()




if __name__ == '__main__':
    startapplication()