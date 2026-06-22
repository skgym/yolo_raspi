# from predict_webcram import frame
import time

from ultralytics import YOLO

# model = YOLO("/home/dubx/Projects/yolov11/model/best2.pt")
model = YOLO("model/best2.pt")


source = "video/2.mp4"  # 2
imgsource = "/home/dubx/Projects/yolov11/video/test1.png"

time1 = time.time()
results = model.predict(source, vid_stride=1, imgsz=640, conf=0.6, save=False, save_txt=True, save_crop=True)
time2 = time.time()
# results = model.predict(imgsource, imgsz=640, conf=0.15, save = True, save_txt=True, save_crop=True, classes=2)#, classes=2


# for i in range(100):
#     results = model.predict(imgsource, imgsz=640, vid_stride =2, conf=0.6, save=True, save_txt=True, save_crop=True)  # , classes=2
#     sleep(2)


# print("total time:", time2-time1)


# phash     215.02 s
# kv cache
