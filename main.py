from detect import AimBot
import cv2
from utils.torch_utils import select_device
from models.common import DetectMultiBackend


cudade = select_device('0')
model = DetectMultiBackend("yolov7.pt",cudade, dnn=False, data=None, fp16=True)
img = cv2.imread('C:\\Users\\hbwindows\\Downloads\\h.png')
dpt = AimBot(cudade, model, 0.6, (640,640))
print(dpt.init_cam())
while True:
    im , detections = dpt.get_detections(source=cv2.cvtColor(dpt.grab_screen(), cv2.COLOR_BGR2RGB))
    cv2.imshow("test.png",im)
    cv2.waitKey(1)
