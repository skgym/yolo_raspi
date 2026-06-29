import argparse
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

from raspi_app.camera_check import (
    LatestFrameCapture,
    build_output_path,
    create_writer,
    parse_source,
    save_crops,
    save_label_txt,
)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--model", default="model/best2.pt", help="YOLO model path")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--project", default="runs", help="output root directory")
    parser.add_argument("--name", default=None, help="output run directory name")
    # 撮影・保存する映像の解像度。YOLO の推論サイズは --imgsz で別に指定する。
    parser.add_argument("--width", type=int, default=1920, help="camera capture width")
    parser.add_argument("--height", type=int, default=1080, help="camera capture height")
    parser.add_argument("--fps", type=float, default=30.0, help="requested camera fps")
    # 推論が遅い環境では、実処理 FPS に近い値へ下げると保存動画が早送りになりにくい。
    parser.add_argument("--output-fps", type=float, default=5.0, help="saved video fps")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--show", action="store_true", help="show preview window")
    parser.add_argument("--no-save", action="store_true", help="do not save annotated mp4")
    parser.add_argument("--no-save-txt", action="store_true", help="do not save label txt files")
    parser.add_argument("--save-crop", action="store_true", help="save cropped detections")
    return parser.parse_args()


def wait_for_first_frame(cap, source):
    # カメラスレッドが最初のフレームを読むまで待つ。
    while True:
        ok, _ = cap.read()

        if ok:
            return

        if not cap.running:
            raise RuntimeError(f"failed to read source: {source}")

        time.sleep(0.01)


def main():
    args = parse_args()
    source = parse_source(args.source)
    if args.name is None:
        args.name = f"camera_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 保存先は camera_check と同じく、リポジトリ直下の runs/<name>/<name>.mp4。
    output_path = build_output_path(args.project, args.name)
    output_dir = output_path.parent

    # Raspberry Pi のカメラ映像を YOLO に直接渡してリアルタイム推論する。
    print(f"load model: {args.model}", flush=True)
    model = YOLO(args.model)

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
    # 送信は別スレッドで行い、通信待ちで推論ループが止まらないようにする。
    sender.start()

    cap = None
    writer = None

    try:
        print(f"open source: {source}", flush=True)
        # camera_check と同じく、裏スレッドでカメラを読み続けて最新フレームだけ使う。
        cap = LatestFrameCapture(source, args.width, args.height, args.fps)
        cap.start()
        wait_for_first_frame(cap, source)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"capture size: {frame_width}x{frame_height}", flush=True)

        if not args.no_save:
            # YOLO の save=True ではなく OpenCV で .mp4 として保存する。
            writer = create_writer(output_path, args.output_fps, (frame_width, frame_height))
            print(f"save: {output_path}", flush=True)

        frame_index = 0

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_index += 1
            # 撮影解像度は高く保ちつつ、推論負荷は --imgsz で調整する。
            result = model.predict(frame, imgsz=args.imgsz, save=False, verbose=False)[0]
            annotated_frame = result.plot()

            # 各フレームの推論結果に、その時点の GPS と補正済み時刻を付けて送信する。
            gps_data = gps_reader.get_current()
            payload = payload_builder.build(result=result, gps_data=gps_data)
            sender.send(payload)

            detection_count = 0 if result.boxes is None else len(result.boxes)
            print(f"frame={frame_index} detections={detection_count}", flush=True)

            if writer is not None:
                writer.write(annotated_frame)

            if not args.no_save_txt:
                # 検知確認用に YOLO 形式に近いラベル txt を保存する。
                save_label_txt(result, output_dir, frame_index)

            if args.save_crop:
                # crop 保存は重めなので、必要なときだけ --save-crop で有効にする。
                save_crops(result, frame, output_dir, frame_index)

            if args.show:
                cv2.imshow("camera_runner", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

    except KeyboardInterrupt:
        print("stopping...")

    finally:
        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

        if args.show:
            cv2.destroyAllWindows()

        # 例外や Ctrl+C で抜ける場合も、送信用スレッドを終了させる。
        sender.stop()


if __name__ == "__main__":
    main()
