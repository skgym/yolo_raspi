from ultralytics import YOLO

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
    # Raspberry Pi のカメラ映像を YOLO に直接渡してリアルタイム推論する。
    model = YOLO("yolo11n.pt")

    # サーバー時刻との差分を先に測り、以降の送信時刻を補正する。
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
        # stream=True にすると、フレームごとの結果を逐次受け取れる。
        results = model.predict(
            source=0,
            stream=True,
            save=False,
            save_txt=False,
            save_crop=False,
        )

        for result in results:
            # 各フレームの推論結果に、その時点の GPS と補正済み時刻を付けて送信する。
            gps_data = gps_reader.get_current()
            payload = payload_builder.build(result=result, gps_data=gps_data)
            sender.send(payload)

    finally:
        # 例外や Ctrl+C で抜ける場合も、送信用スレッドを終了させる。
        sender.stop()


if __name__ == "__main__":
    main()
