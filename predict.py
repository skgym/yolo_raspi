from time import sleep

# from predict_webcram import frame
from ultralytics import YOLO
import time

#model = YOLO("/home/dubx/Projects/yolov11/model/best2.pt")
model = YOLO("model/best2.pt")


source = "video/1.mp4"  #2
imgsource = "video/test3.png"

time1 = time.time()
#results = model.predict(source, vid_stride =1, imgsz=640, conf=0.6, save=False, save_txt=True, save_crop=True)
#time2 = time.time()
results = model.track(
    imgsource, 
    imgsz=640, 
    conf=0.05, 
    save = True, 
    save_txt=True, 
    save_crop=True, 
    classes=None,
    # === トラッキング関連の引数 ===
    persist=True,
    tracker='bytetrack.yaml' # デフォルトのByteTrackトラッカー設定を使用します
    # ==========================
    )#, classes=2
time2 = time.time()

# for i in range():
#     results = model.predict(imgsource, imgsz=640, vid_stride =2, conf=0.6, save=True, save_txt=True, save_crop=True)  # , classes=2
#     sleep(2)
# time2 = time.time()

print("total time:", time2-time1)


# phash     215.02 s
#kv cache