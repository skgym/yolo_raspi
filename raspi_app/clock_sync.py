"""Raspberry PiとIsaac Sim側PCの時計のずれを簡易的に補正する。"""

import time
import requests


class ClockSynchronizer:
    """サーバー時刻との差分を推定し、送信時刻を補正するクラス。"""

    def __init__(self, time_url, samples=5, timeout=2.0):
        """時刻API、試行回数、通信タイムアウトを設定する。"""
        self.time_url = time_url
        self.samples = samples
        self.timeout = timeout
        # time.time() に足す補正値。同期に失敗した場合は 0 のまま使う。
        self.time_offset = 0.0

    def synchronize(self):
        """複数回のHTTP応答からサーバーとの時刻差を推定する。"""
        offsets = []

        for _ in range(self.samples):
            try:
                # 簡易的な NTP と同じ考え方で、往復時間の半分を通信遅延として扱う。
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
                # 一部の試行に失敗しても、成功したサンプルだけで補正値を計算する。
                print(f"[ClockSynchronizer] failed: {e}")

        if offsets:
            # 複数回測って平均することで、通信遅延のばらつきを少しならす。
            self.time_offset = sum(offsets) / len(offsets)

        return self.time_offset

    def now(self):
        """推定した時刻差を反映した現在時刻を秒単位で返す。"""
        # アプリ内ではこのメソッドを使い、補正済みの現在時刻を取得する。
        return time.time() + self.time_offset

    def now_ms(self):
        """推定した時刻差を反映した現在時刻を整数ミリ秒で返す。"""
        # JSON とログで扱いやすいよう、補正済み時刻を整数のミリ秒にする。
        return int(round(self.now() * 1000))
