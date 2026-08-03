"""raspi_appの初期ファイル一式を作成するためのセットアップスクリプト。"""

from pathlib import Path


# 実行したディレクトリをプロジェクト直下とみなし、raspi_appをその中へ作る。
BASE_DIR = Path.cwd()
APP_DIR = BASE_DIR / "raspi_app"


# キーが作成先、値が初期コードを表す。既存ファイルは後段で上書きしない。
FILES = {
    "raspi_app/__init__.py": '''"""Raspberry Pi上でYOLO推論結果を送信するアプリ群。"""\n''',
    "raspi_app/config.py": '''"""Isaac Sim側サーバーへの接続と送信キューに関する共通設定。"""

# 検知結果の送信先。時計はNerveNet構築後にOSのNTP機能で同期する。
ENDPOINT_URL = "http://192.168.0.122:40000/endpoint"

# 通信待ち時間と未送信キュー数。
SEND_TIMEOUT = 2.0
QUEUE_SIZE = 1
''',
    "raspi_app/gps_reader.py": '''"""GPS情報を推論結果へ追加するための読み取り処理。"""


class GpsReader:
    """GPSの現在位置を共通形式で返す。"""

    def get_current(self):
        """現在位置を緯度・経度の辞書で返す。"""
        # TODO: 実機ではここをGPSモジュールからの読み取り処理に置き換える。
        return {
            "lat": None,
            "lon": None,
        }
''',
    "raspi_app/payload_builder.py": '''"""YOLO推論結果を送信用の辞書へ変換する。"""

import time


class DetectionPayloadBuilder:
    """フレーム単位の時刻、GPS、検知結果を送信用にまとめる。"""

    def __init__(self):
        """送信先で処理順を確認するためのフレーム連番を初期化する。"""
        self.frame_id = 0

    def build(self, result, gps_data=None):
        """1フレーム分の送信データを作る。"""
        self.frame_id += 1

        payload = {
            "frame_id": self.frame_id,
            # OSのUnix時刻をms単位で記録する。NTP同期はアプリ外で行う。
            "timestamp_send": time.time_ns() // 1_000_000,
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
        """YOLOの前処理・推論・後処理時間を合計する。"""
        if not result.speed:
            return 0.0

        total_ms = 0.0

        for value in result.speed.values():
            if value is not None:
                total_ms += value

        return round(total_ms, 3)

    def _build_detection(self, result, box):
        """1つの検知枠をJSON化可能な辞書へ変換する。"""
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
    "raspi_app/result_sender.py": '''"""推論を止めずに結果を送るバックグラウンド送信処理。"""

import copy
import queue
import threading
import requests


class BackgroundResultSender:
    """最新の推論結果を優先しながら別スレッドでHTTP送信する。"""

    def __init__(self, endpoint_url, queue_size=1, timeout=2.0):
        """送信先、待ち行列サイズ、通信タイムアウトを設定する。"""
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.thread = None

    def start(self):
        """HTTP送信用のバックグラウンドスレッドを開始する。"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def send(self, payload):
        """送信データを待ち行列へ入れ、推論処理へすぐ戻る。"""
        payload = copy.deepcopy(payload)

        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            # 通信が遅い場合は古い未送信結果を捨て、最新フレームを優先する。
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
        """待ち行列からデータを取り出してサーバーへ送信する。"""
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
        """送信スレッドへ終了を通知し、短時間だけ終了を待つ。"""
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2.0)
''',
    "raspi_app/camera_runner.py": '''"""カメラ映像をYOLOで推論してサーバーへ送信する。"""

from ultralytics import YOLO

from raspi_app.config import (
    ENDPOINT_URL,
    QUEUE_SIZE,
    SEND_TIMEOUT,
)
from raspi_app.gps_reader import GpsReader
from raspi_app.payload_builder import DetectionPayloadBuilder
from raspi_app.result_sender import BackgroundResultSender


def main():
    """モデル、OS時刻、GPS、非同期送信を組み合わせて推論を実行する。"""
    # YOLOモデルを読み込み、カメラからストリーム形式で推論する。
    model = YOLO("yolo11n.pt")

    # PCとの時刻同期はアプリ外のNTPに任せる。
    gps_reader = GpsReader()
    payload_builder = DetectionPayloadBuilder()

    sender = BackgroundResultSender(
        endpoint_url=ENDPOINT_URL,
        queue_size=QUEUE_SIZE,
        timeout=SEND_TIMEOUT,
    )
    sender.start()

    try:
        # stream=Trueにより、結果を1フレームずつ受け取る。
        results = model.predict(
            source=0,
            stream=True,
            save=False,
            save_txt=False,
            save_crop=False,
        )

        for result in results:
            # フレームごとにGPSと検知結果をまとめ、送信キューへ入れる。
            gps_data = gps_reader.get_current()
            payload = payload_builder.build(result=result, gps_data=gps_data)
            sender.send(payload)

    finally:
        # 例外やCtrl+Cで終了する場合も送信スレッドを停止する。
        sender.stop()


if __name__ == "__main__":
    main()
''',
}


def main():
    """raspi_appディレクトリを作り、存在しない初期ファイルだけを書き出す。"""
    APP_DIR.mkdir(parents=True, exist_ok=True)

    for relative_path, content in FILES.items():
        path = BASE_DIR / relative_path

        # ユーザーが編集済みのコードを誤って消さないため、既存ファイルは変更しない。
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
