from ultralytics import YOLO

model = YOLO("model/best2.pt")

source = "video/2.mp4" 

results = model.track(
    source, 
    imgsz=640, 
    conf=0.5, 
    save=True,
    save_crop=True, 
    classes=None,
    persist=True,
    tracker='bytetrack.yaml',
    stream=True 
)

for result in results:
    pass 