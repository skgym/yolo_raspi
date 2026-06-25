class DetectionPayloadBuilder:
    """YOLO の Result オブジェクトを送信用 JSON payload に変換する。"""

    def __init__(self, clock):
        self.clock = clock
        # 送信先でフレームの順序を追えるよう、アプリ側で連番を振る。
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
            # 検出が 0 件でも、時刻や GPS を含む空の payload として送る。
            return payload

        for box in result.boxes:
            payload["detections"].append(self._build_detection(result, box))

        return payload

    def _get_processing_time(self, result):
        if not result.speed:
            return 0.0

        # Ultralytics の speed は ms 単位なので、送信用には秒へ変換する。
        total_ms = 0.0

        for value in result.speed.values():
            if value is not None:
                total_ms += value

        return total_ms / 1000.0

    def _build_detection(self, result, box):
        class_id = int(box.cls[0])
        class_name = result.names[class_id]

        # bbox は [x1, y1, x2, y2] のピクセル座標。tolist() で JSON 化できる形にする。
        detection = {
            "class_id": class_id,
            "class_name": class_name,
            "bbox": box.xyxy[0].tolist(),
        }

        if box.conf is not None:
            detection["confidence"] = float(box.conf[0])

        if box.id is not None:
            # tracker を使った場合だけ track_id が入るため、存在するときだけ載せる。
            detection["track_id"] = int(box.id[0])

        return detection
