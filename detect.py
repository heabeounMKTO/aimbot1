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
from simple_pid import PID
from mouse_driver.MouseMove import mouse_move
from pynput.mouse import Button, Listener


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class AimBot:
    def __init__(self, device, model, conf, imgsz):
        self.model = model
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.max_step_dis = 300  # max movement for each iter (smoothness).
        self.max_pid_dis = (
            20  # enanble pid control if target distance is smaller than this distance
        )
        self.smooth = 0.8 * 1920 / 1920  # moving smoothness 1920 = screen
        self.pos_factor = 0.5  # position factor, moves shit close to head.
        self.detect_length = imgsz[0]
        self.detect_center_x, self.detect_center_y = (
            self.detect_length // 2,
            self.detect_length // 2,
        )
        # magic numbers for now, init mouse related:
        self.pidx_kp = 1.2
        self.pidx_kd = 3.51
        self.pidx_ki = 0.0
        self.pidy_kp = 1.22
        self.pidy_kd = 0.24
        self.pidy_ki = 0.0
        
        #lock param and mouse listeners
        self.locking = False
        self.auto_lock = True
        self.mouse_button_1 = "x2"
        self.mouse_button_2 = "x2"
        self.auto_lock_button = "x1"
        listener = Listener(on_click=self.on_click)
        listener.start()

    def grab_screen(self):
        return cv2.cvtColor(
            np.asarray(self.camera.grab(self.region)), cv2.COLOR_BGR2RGB
        )

    def init_cam(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.top, self.left = math.floor(self.screen_height / 2) - math.floor(
            self.detect_length / 2
        ), math.floor(self.screen_width / 2) - math.floor(self.detect_length / 2)
        self.camera = mss()
        self.region = {
            "top": self.top,
            "left": self.left,
            "width": self.detect_length,
            "height": self.detect_length,
        }

    def on_click(self, x, y, button, pressed):
        if button == getattr(Button, self.auto_lock_button) and pressed:
            if self.auto_lock:
                self.auto_lock = False
                print("---control off----")
            else:
                self.auto_lock = True
                print("---control on----")
        if (
            button
            in [
                getattr(Button, self.mouse_button_1),
                getattr(Button, self.mouse_button_2),
            ]
            and self.auto_lock
        ):
            if pressed:
                self.locking = True
                print("locking ON")
            else:
                self.locking = False
                print("locking OFF")
        print(f"button {button.name} pressed\n")

    def init_mouse(self):
        self.pidx = PID(
            self.pidx_kp,
            self.pidx_kd,
            self.pidx_ki,
            setpoint=0,
            sample_time=0.001,
        )
        self.pidy = PID(
            self.pidy_kp,
            self.pidy_kd,
            self.pidy_ki,
            setpoint=0,
            sample_time=0.001,
        )
        self.pidx(0), self.pidy(0)

    def get_detections(self, source, half=False, iou_thres=0.5, max_det=100):
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
        model.warmup(
            imgsz=(1, 3, *self.imgsz)
        )  # runs the model through np.zeroes to warmup
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
                    # xywh = (
                    #     (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                    #     .view(-1)
                    #     .tolist()
                    # )
                    md = self.move_dis(xyxy)
                    detection_dict = {
                        "move_dist": md,
                        "avg_x": (xyxy[0] + xyxy[2]) / 2,
                        "avg_y": (xyxy[1] + xyxy[3]) / 2,
                        "cls": cls,
                    }
                    detections.append(detection_dict)
                    plot_one_box(
                        x=xyxy,
                        img=im0,
                        color=(255, 0, 0),
                        label=f"dist: {md:.2f} \n conf: {conf:.2f}",
                        line_thickness=1,
                    )
        self.sorted_move_targets = sorted(detections, key=lambda x: (x["move_dist"]))
        return im0, self.sorted_move_targets

    def lock_target(self):
        if len(self.sorted_move_targets) > 0 and self.locking:
            move_rel_x, move_rel_y, move_dis = self.get_move_dist()
            mouse_move(move_rel_x, move_rel_y)
        self.pidx(0), self.pidy(0)

    def get_move_dist(self):
        try:
            nearest_target = min(
                self.sorted_move_targets, key=lambda x: (x["move_dist"])
            )
            target_x, target_y, move_dis = (
                nearest_target["avg_x"],
                nearest_target["avg_y"],
                nearest_target["move_dist"],
            )
            # compute relative movement distance to target.
            move_relx = (target_x - self.detect_center_x) * self.smooth
            move_rely = (target_y - self.detect_center_y) * self.smooth
            if move_dis >= self.max_step_dis:
                move_relx = move_relx / move_dis * self.max_step_dis
                move_rely = move_rely / move_dis * self.max_step_dis
            elif move_dis <= self.max_pid_dis:
                move_relx = self.pidx(
                    math.atan2(-move_relx, self.detect_length) * self.detect_length
                )
                move_rely = self.pidy(
                    math.atan2(-move_rely, self.detect_length) * self.detect_length
                )
            return float(move_relx), float(move_rely), float(move_dis)
        except ValueError:
            pass

    def move_dis(self, box):
        x1, y1, x2, y2 = box
        target_x, target_y = (x1 + x2) / 2, (y1 + y2) / 2 - self.pos_factor * (y2 - y1)
        move_dis = (
            (target_x - self.detect_center_x) ** 2
            + (target_y - self.detect_center_y) ** 2
        ) ** (1 / 2)
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
