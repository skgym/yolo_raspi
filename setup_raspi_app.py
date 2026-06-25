from pathlib import Path


BASE_DIR = Path.cwd()
APP_DIR = BASE_DIR / "raspi_app"


FILES = {
    "raspi_app/__init__.py": '''"""Raspberry Pi application helpers for YOLO result sending."""\n''',
    "raspi_app/config.py": '''TIME_URL = "http://192.168.0.122:40000/time"
ENDPOINT_URL = "http://192.168.0.122:40000/endpoint"

SEND_TIMEOUT = 2.0
QUEUE_SIZE = 1
CLOCK_SYNC_SAMPLES = 5
''',
    "raspi_app/clock_sync.py": '''import time
import requests


class ClockSynchronizer:
    def __init__(self, time_url, samples=5, timeout=2.0):
        self.time_url = time_url
        self.samples = samples
        self.timeout = timeout
        self.time_offset = 0.0

    def synchronize(self):
        offsets = []

        for _ in range(self.samples):
            try:
                t1 = time.time()
                response = requests.get(self.time_url, timeout=self.timeout)
                response.raise_for_status()
                t4 = time.time()

                server_time = response.json()["server_time"]
                rtt = t4 - t1
                latency = rtt / 2
                offset = server_time - (t1 + latency)
                offsets.append(offset)

            except Exception as e:
                print(f"[ClockSynchronizer] failed: {e}")

        if offsets:
            self.time_offset = sum(offsets) / len(offsets)

        return self.time_offset

    def now(self):
        return time.time() + self.time_offset
''',
    "raspi_app/gps_reader.py": '''class GpsReader:
    def get_current(self):
        # TODO: Replace this with the actual GPS module reading.
        return {
            "lat": None,
            "lon": None,
        }
''',
    "raspi_app/payload_builder.py": '''class DetectionPayloadBuilder:
    def __init__(self, clock):
        self.clock = clock
        self.frame_id = 0

    def build(self, result, gps_data=None):
        self.frame_id += 1

        payload = {
            "frame_id": self.frame_id,
            "timestamp_send": self.clock.now(),
            "t_proc": self._get_processing_time(result),
            "gps": gps_data,
            "detections": [],
        }

        if result.boxes is None:
            return payload

        for box in result.boxes:
            payload["detections"].append(self._build_detection(result, box))

        return payload

    def _get_processing_time(self, result):
        if not result.speed:
            return 0.0

        total_ms = 0.0

        for value in result.speed.values():
            if value is not None:
                total_ms += value

        return total_ms / 1000.0

    def _build_detection(self, result, box):
        class_id = int(box.cls[0])
        class_name = result.names[class_id]

        detection = {
            "class_id": class_id,
            "class_name": class_name,
            "bbox": box.xyxy[0].tolist(),
        }

        if box.conf is not None:
            detection["confidence"] = float(box.conf[0])

        if box.id is not None:
            detection["track_id"] = int(box.id[0])

        return detection
''',
    "raspi_app/result_sender.py": '''import copy
import queue
import threading
import requests


class BackgroundResultSender:
    def __init__(self, endpoint_url, queue_size=1, timeout=2.0):
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def send(self, payload):
        payload = copy.deepcopy(payload)

        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            # Keep the latest frame result and drop an older unsent one.
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                pass

            try:
                self.queue.put_nowait(payload)
            except queue.Full:
                pass

    def _worker(self):
        session = requests.Session()

        while self.running:
            try:
                payload = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                response = session.post(
                    self.endpoint_url,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code != 200:
                    print(f"[BackgroundResultSender] status={response.status_code}")

            except requests.RequestException as e:
                print(f"[BackgroundResultSender] failed: {e}")

            finally:
                self.queue.task_done()

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2.0)
''',
    "raspi_app/camera_runner.py": '''from ultralytics import YOLO

from raspi_app.config import (
    CLOCK_SYNC_SAMPLES,
    ENDPOINT_URL,
    QUEUE_SIZE,
    SEND_TIMEOUT,
    TIME_URL,
)
from raspi_app.clock_sync import ClockSynchronizer
from raspi_app.gps_reader import GpsReader
from raspi_app.payload_builder import DetectionPayloadBuilder
from raspi_app.result_sender import BackgroundResultSender


def main():
    model = YOLO("yolo11n.pt")

    clock = ClockSynchronizer(
        time_url=TIME_URL,
        samples=CLOCK_SYNC_SAMPLES,
        timeout=SEND_TIMEOUT,
    )
    clock.synchronize()

    gps_reader = GpsReader()
    payload_builder = DetectionPayloadBuilder(clock)

    sender = BackgroundResultSender(
        endpoint_url=ENDPOINT_URL,
        queue_size=QUEUE_SIZE,
        timeout=SEND_TIMEOUT,
    )
    sender.start()

    try:
        results = model.predict(
            source=0,
            stream=True,
            save=False,
            save_txt=False,
            save_crop=False,
        )

        for result in results:
            gps_data = gps_reader.get_current()
            payload = payload_builder.build(result=result, gps_data=gps_data)
            sender.send(payload)

    finally:
        sender.stop()


if __name__ == "__main__":
    main()
''',
}


def main():
    APP_DIR.mkdir(parents=True, exist_ok=True)

    for relative_path, content in FILES.items():
        path = BASE_DIR / relative_path

        if path.exists():
            print(f"skip: {relative_path} already exists")
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"created: {relative_path}")

    print()
    print("done.")
    print("Run with:")
    print("  python -m raspi_app.camera_runner")


if __name__ == "__main__":
    main()
