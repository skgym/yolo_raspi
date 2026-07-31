"""推論を止めずにIsaac SimへJSONを送信するバックグラウンド送信処理。"""

import copy
import queue
import threading
import time
import requests


def build_isaac_payload(payload):
    """フレーム形式の辞書を、Isaac Simが期待する検知結果のリストへ変換する。"""
    detections = payload.get("detections", [])

    # 検出が 0 件でも、Isaac Sim に送る JSON は必ずリストにする。
    if not detections:
        return []

    isaac_payload = []

    for detection in detections:
        item = copy.deepcopy(detection)
        # フレーム単位の情報も、Isaac Sim 側で必要になったとき使えるよう各検出へ持たせる。
        item["frame_id"] = payload.get("frame_id")
        item["timestamp_send"] = payload.get("timestamp_send")
        item["t_proc"] = payload.get("t_proc")
        item["gps"] = payload.get("gps")
        isaac_payload.append(item)

    return isaac_payload


class BackgroundResultSender:
    """推論ループを止めないように、結果送信を別スレッドで行う。"""

    def __init__(self, endpoint_url, queue_size=1, timeout=2.0, run_logger=None):
        """送信先、待ち行列サイズ、タイムアウト、任意のログ保存先を設定する。"""
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        self.run_logger = run_logger
        # キューを小さく保ち、通信が遅いときは古いフレームを捨てて遅延を増やさない。
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
        """最新の送信データを待ち行列へ入れ、呼び出し元へすぐ戻る。"""
        # 呼び出し元が後から payload を変更しても送信内容に影響しないようコピーする。
        payload = copy.deepcopy(payload)
        # Isaac Sim 側は list の各要素に class_name がある前提で読む。
        frame_id = payload.get("frame_id")
        payload = build_isaac_payload(payload)
        queued_item = (frame_id, payload)

        try:
            self.queue.put_nowait(queued_item)
        except queue.Full:
            # 通信遅延を蓄積させないため、未送信の古い結果を捨てて最新結果を残す。
            try:
                dropped_frame_id, dropped_payload = self.queue.get_nowait()
                self.queue.task_done()

                if self.run_logger is not None:
                    self.run_logger.log_network(
                        frame_id=dropped_frame_id,
                        event="dropped",
                        payload_items=len(dropped_payload),
                    )
            except queue.Empty:
                pass

            try:
                self.queue.put_nowait(queued_item)
            except queue.Full:
                if self.run_logger is not None:
                    self.run_logger.log_network(
                        frame_id=frame_id,
                        event="dropped",
                        payload_items=len(payload),
                    )

    def _worker(self):
        """キューから結果を取り出し、送信成否と通信時間を記録する。"""
        # Session を使い回すことで、連続 POST の接続コストを下げる。
        session = requests.Session()

        while self.running or not self.queue.empty():
            try:
                frame_id, payload = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            started_at = time.perf_counter()

            try:
                # サーバーには payload を JSON として送る。
                response = session.post(
                    self.endpoint_url,
                    json=payload,
                    timeout=self.timeout,
                )
                elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)

                if response.status_code != 200:
                    print(f"[BackgroundResultSender] status={response.status_code}")

                if self.run_logger is not None:
                    event = "sent" if response.status_code == 200 else "failed"
                    self.run_logger.log_network(
                        frame_id=frame_id,
                        event=event,
                        status_code=response.status_code,
                        elapsed_ms=elapsed_ms,
                        payload_items=len(payload),
                        payload=payload,
                    )

            except requests.RequestException as e:
                print(f"[BackgroundResultSender] failed: {e}")
                elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)

                if self.run_logger is not None:
                    self.run_logger.log_network(
                        frame_id=frame_id,
                        event="failed",
                        elapsed_ms=elapsed_ms,
                        payload_items=len(payload),
                        error=str(e),
                        payload=payload,
                    )

            finally:
                # 成功・失敗に関係なく、この payload の処理は完了扱いにする。
                self.queue.task_done()

    def stop(self):
        """新規受付を止め、キューに残った最後の結果の送信終了を待つ。"""
        self.running = False

        if self.thread is not None:
            # daemon thread だが、短時間だけ待って終了処理をきれいにする。
            self.thread.join(timeout=self.timeout + 2.0)
