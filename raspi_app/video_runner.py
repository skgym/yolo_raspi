"""保存済み動画で、カメラ実機と同じ推論・送信形式を確認する。"""

import argparse
import json

from ultralytics import YOLO

from raspi_app.config import ENDPOINT_URL, QUEUE_SIZE, SEND_TIMEOUT
from raspi_app.gps_reader import GpsReader
from raspi_app.payload_builder import DetectionPayloadBuilder
from raspi_app.result_sender import BackgroundResultSender, build_isaac_payload


def parse_args():
    """入力動画、モデル、送信有無、処理フレーム数を受け取る。"""
    # 動画ファイルで送信処理を試すための実行オプションを受け取る。
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="input video path")
    parser.add_argument("--model", default="model/best2.pt", help="YOLO model path")
    parser.add_argument("--no-send", action="store_true", help="print payload only")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per frame")
    return parser.parse_args()


def main():
    """動画を1フレームずつ推論し、結果を表示またはIsaac Simへ送信する。"""
    args = parse_args()

    # 指定された動画を YOLO に入力し、Raspberry Pi 実機のカメラなしで挙動確認する。
    model = YOLO(args.model)

    # 時刻はOSのUnix時刻を使い、PCとの同期はアプリ外のNTPに任せる。
    gps_reader = GpsReader()
    payload_builder = DetectionPayloadBuilder()

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
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
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
