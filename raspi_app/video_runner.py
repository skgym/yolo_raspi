# raspi_app/video_runner.py

import argparse
import json

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
from raspi_app.result_sender import BackgroundResultSender, build_isaac_payload


def parse_args():
    # 動画ファイルで送信処理を試すための実行オプションを受け取る。
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="input video path")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path")
    parser.add_argument("--no-send", action="store_true", help="print payload only")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means unlimited")
    return parser.parse_args()


def main():
    args = parse_args()

    # 指定された動画を YOLO に入力し、Raspberry Pi 実機のカメラなしで挙動確認する。
    model = YOLO(args.model)

    # 実機実行時と同じ payload 形式になるよう、時刻同期と GPS 読み取り部品を使う。
    clock = ClockSynchronizer(
        time_url=TIME_URL,
        samples=CLOCK_SYNC_SAMPLES,
        timeout=SEND_TIMEOUT,
    )
    clock.synchronize()

    gps_reader = GpsReader()
    payload_builder = DetectionPayloadBuilder(clock)

    sender = None

    if not args.no_send:
        # 通信処理は別スレッドに逃がし、推論ループをできるだけ止めない。
        sender = BackgroundResultSender(
            endpoint_url=ENDPOINT_URL,
            queue_size=QUEUE_SIZE,
            timeout=SEND_TIMEOUT,
        )
        sender.start()

    try:
        # stream=True により、動画全体を待たずに 1 フレームずつ処理できる。
        results = model.predict(
            source=args.video,
            stream=True,
            save=False,
            save_txt=False,
            save_crop=False,
            verbose=False,
        )

        for frame_index, result in enumerate(results, start=1):
            # 実機と同じ流れで、フレームごとに GPS 情報を payload に含める。
            gps_data = gps_reader.get_current()

            payload = payload_builder.build(
                result=result,
                gps_data=gps_data,
            )

            if args.no_send:
                # 実送信時と同じく、Isaac Sim 向け JSON は必ずリストで包む。
                print(json.dumps(build_isaac_payload(payload), ensure_ascii=False))
            else:
                sender.send(payload)

            # デバッグ時に短い区間だけ処理できるよう、任意でフレーム数を制限する。
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

    finally:
        # --no-send の場合 sender は作らないため、存在するときだけ停止する。
        if sender is not None:
            sender.stop()


if __name__ == "__main__":
    main()
