"""Ultralyticsの推論結果をIsaac Sim送信用の辞書へ変換する。"""

import time


class DetectionPayloadBuilder:
    """YOLO の Result オブジェクトを送信用 JSON payload に変換する。"""

    def __init__(self):
        """送信先で処理順を確認するためのフレーム連番を初期化する。"""
        # 送信先でフレームの順序を追えるよう、アプリ側で連番を振る。
        self.frame_id = 0

    def build(self, result, gps_data=None):
        """1フレーム分の時刻、処理時間、GPS、検知結果をまとめる。"""
        self.frame_id += 1

        payload = {
            "frame_id": self.frame_id,
            # OSのUnix時刻をms単位で記録する。時計のNTP同期はアプリ外で行う。
            "timestamp_send": time.time_ns() // 1_000_000,
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
        """前処理・推論・後処理にかかった合計時間をミリ秒で返す。"""
        if not result.speed:
            return 0.0

        # Ultralytics の speed は元から ms 単位なので、合計値も ms のまま返す。
        total_ms = 0.0

        for value in result.speed.values():
            if value is not None:
                total_ms += value

        return round(total_ms, 3)

    def _build_detection(self, result, box):
        """YOLOの1つの検知枠をJSON化できる値だけで構成する。"""
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
