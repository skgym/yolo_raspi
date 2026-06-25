import copy
import queue
import threading
import requests


def build_isaac_payload(payload):
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

    def __init__(self, endpoint_url, queue_size=1, timeout=2.0):
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        # キューを小さく保ち、通信が遅いときは古いフレームを捨てて遅延を増やさない。
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
        # 呼び出し元が後から payload を変更しても送信内容に影響しないようコピーする。
        payload = copy.deepcopy(payload)
        # Isaac Sim 側は list の各要素に class_name がある前提で読む。
        payload = build_isaac_payload(payload)

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
        # Session を使い回すことで、連続 POST の接続コストを下げる。
        session = requests.Session()

        while self.running:
            try:
                payload = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                # サーバーには payload を JSON として送る。
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
                # 成功・失敗に関係なく、この payload の処理は完了扱いにする。
                self.queue.task_done()

    def stop(self):
        self.running = False

        if self.thread is not None:
            # daemon thread だが、短時間だけ待って終了処理をきれいにする。
            self.thread.join(timeout=2.0)
