from detect import AimBot
import cv2
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from pprint import pprint

cudade = select_device('0')
model = DetectMultiBackend("yolov7.pt",cudade, dnn=False, data=None, fp16=True)
dpt = AimBot(cudade, model, 0.75, (640,640))
dpt.init_cam()
dpt.init_mouse()
while True:
    im , detections = dpt.get_detections(source=cv2.cvtColor(dpt.grab_screen(), cv2.COLOR_BGR2RGB))
    pprint(dpt.get_move_dist())
    cv2.imshow("test.png",im)
    cv2.waitKey(1)
