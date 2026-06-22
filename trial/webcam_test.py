import cv2
from picamera2 import Picamera2

picam2 = Picamera2

picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configur("preview")

picam2.start()

try:
    while True:

        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Rasberry Pi Camera", frame_bgr)
        if cv2.waitkey(1) == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyALLWindows()



