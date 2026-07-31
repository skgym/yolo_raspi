"""Isaac Sim側サーバーへの接続と送信キューに関する共通設定。"""

# Raspberry Pi から接続するサーバーの時刻同期 API と結果送信 API。
SERVER_HOST = "172.16.3.128"
SERVER_PORT = 40000
TIME_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/time"
ENDPOINT_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/endpoint"

# 通信や送信待ちの設定。QUEUE_SIZE=1 は最新フレームを優先するための値。
SEND_TIMEOUT = 2.0
QUEUE_SIZE = 1
CLOCK_SYNC_SAMPLES = 5
