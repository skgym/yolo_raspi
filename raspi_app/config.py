# Raspberry Pi から接続するサーバーの時刻同期 API と結果送信 API。
TIME_URL = "http://192.168.0.122:40000/time"
ENDPOINT_URL = "http://192.168.0.122:40000/endpoint"

# 通信や送信待ちの設定。QUEUE_SIZE=1 は最新フレームを優先するための値。
SEND_TIMEOUT = 2.0
QUEUE_SIZE = 1
CLOCK_SYNC_SAMPLES = 5
