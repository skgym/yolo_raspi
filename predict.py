import time
from ultralytics import YOLO
from picamera2 import Picamera2
from pathlib import Path


model = YOLO("model/best2.pt")

picam2 = Picamera2()

config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

try:
    while True:

        frame = picam2.capture_array()

        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.6,
            save=False,
            stream=True
        )

        for result in results:
            if len(result.boxes) > 0:
                print(f"Detected: {len(result.boxes)} objects")

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    picam2.stop()
    picam2.close()
