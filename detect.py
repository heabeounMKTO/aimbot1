import numpy as np
import torch
from utils.general import LOGGER, non_max_suppression, scale_coords, xyxy2xywh
from utils.augmentations import letterbox
from utils.torch_utils import select_device, time_sync
from pprint import pprint
import time
from mss.windows import MSS as mss
from multiprocessing import Process, Queue 
import pyautogui
import math
import cv2
from numpy import random

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

class AimBot:
    def __init__(self, device, model, conf, imgsz):
        self.model = model
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.pos_factor = 0.5 #position factor, moves shit close to head.
        self.detect_length = imgsz[0]
        self.detect_center_x , self.detect_center_y = self.detect_length//2 , self.detect_length//2 
    def grab_screen(self):
        return cv2.cvtColor(np.asarray(self.camera.grab(self.region)), cv2.COLOR_BGR2RGB)
     
    def init_cam(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.top , self.left = math.floor(self.screen_height/2) - math.floor(self.detect_length/2), math.floor(self.screen_width/2) - math.floor(self.detect_length/2)
        self.camera = mss()
        self.region = {"top": self.top, "left": self.left, "width": self.detect_length, "height": self.detect_length}
    
    def get_detections(self, source,half=False, iou_thres=0.5, max_det=100):
        """
        generic detection function, spits out raw; unsorted detections,
        to be used with a detect processing function
        """
        
        model = self.model
        conf_thres = self.conf
        device = select_device(self.device)

        # pre-processing
        im0s = source
        im = self.preProc(im0s, imgsz=self.imgsz)

        stride, names, pt = model.stride, model.names, model.pt
        # model.warmup(imgsz=(1,3,*imgsz)) #runs the model through np.zeroes to warmup
        # seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
        # t1 = time_sync()

        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255  # normalize image from 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]
        # t2 = time_sync()
        # dt[0] += t2 - t1
        
        # dont calculate gradients? it causes memory leak.
        # https://github.com/WongKinYiu/yolov7/commit/072f76c72c641c7a1ee482e39f604f6f8ef7ee92
        with torch.no_grad(): 
            pred = model(im, augment=False, visualize=False)
        # t3 = time_sync()
        # dt[1] += t3 - t2

        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=max_det
        )
        # dt[2] += time_sync() - t3
        # process detections
        detections = []
        for det in pred:
            im0 = im0s.copy()
            # seen += 1
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # seen += 1
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det[:, :6]):
                    xywh = (
                        (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                        .view(-1)
                        .tolist()
                    )
                    md = self.move_dis(xyxy)
                    print(md) 
                    detections.append(xyxy)
                    plot_one_box(x=xyxy,img=im0, color=(255,0,0), label=f'{conf:.2f}', line_thickness=1 )
        return im0 ,detections

    def get_move_dis(self, target_sort_list):
        target_info = min(target_sort_list, key=lambda x: (x['label'], x['move_dis']))
    
    def move_dis(self, box):
        x1, y1, x2, y2 = box
        target_x, target_y = (x1 + x2) / 2, (y1 + y2) / 2 - self.pos_factor * (y2 - y1)
        move_dis = ((target_x - self.detect_center_x) ** 2 + (target_y - self.detect_center_y) ** 2) ** (1 / 2)
        # Sort the list by label and then by distance
        return move_dis 
     
    def preProc(self, im0s, imgsz=640):
        """
        pre-process image into letterbox of specified size
        """
        # print("preproc imgsz", imgsz)
        im = letterbox(im0s, new_shape=imgsz, stride=32, auto=True)
        im = np.array(im[0], dtype=np.uint8)
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        return im





