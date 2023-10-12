from detect import AimBot
import cv2
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from pprint import pprint
import os
from tkinter import *


def init_device(model):
    cudade = select_device("0")
    model = DetectMultiBackend(model, cudade, dnn=False, data=None, fp16=True)
    dpt = AimBot(cudade, model, 0.75, (640, 640))
    return dpt

def start_cheat(dpt):
    dpt.init_cam()
    dpt.init_mouse()
    while True:
        im, detections = dpt.get_detections(
            source=cv2.cvtColor(dpt.grab_screen(), cv2.COLOR_BGR2RGB)
        )
        dpt.lock_target()
        print(f"debug nearest target: {dpt.get_move_dist()}", end='\r')
        cv2.imshow("Preview_Panel", im)
        cv2.waitKey(1)

window = Tk()
window.resizable()
window.title("heabeoun's aimbot client")
window.geometry('640x480')

load_model = "yolov7.pt"
loading_status = StringVar()
loading_status.set("Pung load jam tic")
ayylmao = Label(window, textvariable=loading_status)
ayylmao.pack(side=TOP)
dpt = init_device(load_model)
loading_status.set(f"model loaded {load_model}")
start_cheat(dpt)

window.mainloop()