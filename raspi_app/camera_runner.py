"""カメラの最新フレームをYOLOで推論し、記録しながらIsaac Simへ送信する。"""

import argparse
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

from raspi_app.camera_check import (
    LatestFrameCapture,
    build_output_path,
    create_writer,
    get_capture_fourcc,
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
from raspi_app.run_logger import RunLogger, epoch_ms


def parse_args():
    """撮影、推論、保存、負荷計測に使用する実行オプションを定義する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--model", default="model/best2.pt", help="YOLO model path")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--project", default="runs", help="output root directory")
    parser.add_argument("--name", default=None, help="output run directory name")
    # 撮影・保存する映像の解像度。YOLO の推論サイズは --imgsz で別に指定する。
    parser.add_argument("--width", type=int, default=640, help="camera capture width")
    parser.add_argument("--height", type=int, default=480, help="camera capture height")
    parser.add_argument("--fps", type=float, default=60.0, help="requested camera fps")
    # 推論が遅い環境では、実処理 FPS に近い値へ下げると保存動画が早送りになりにくい。
    parser.add_argument("--output-fps", type=float, default=30.0, help="saved video fps")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per frame")
    parser.add_argument("--show", action="store_true", help="show preview window")
    parser.add_argument(
        "--no-send",
        action="store_true",
        help="do not synchronize clock or send results",
    )
    parser.add_argument("--no-save", action="store_true", help="do not save annotated mp4")
    parser.add_argument("--no-save-txt", action="store_true", help="do not save label txt files")
    parser.add_argument("--save-crop", action="store_true", help="save cropped detections")
    parser.add_argument(
        "--system-log-interval",
        type=float,
        default=1.0,
        help="seconds between system metric samples",
    )
    return parser.parse_args()


def wait_for_first_frame(cap, source):
    """読み取りスレッドが最初のフレームを取得するまで待機する。"""
    # カメラスレッドが最初のフレームを読むまで待つ。
    while True:
        ok, _ = cap.read()

        if ok:
            return

        if not cap.running:
            raise RuntimeError(f"failed to read source: {source}")

        time.sleep(0.01)


def main():
    """カメラ撮影、推論、ログ保存、結果送信をまとめて実行する。"""
    args = parse_args()
    source = parse_source(args.source)
    if args.name is None:
        # 保存フォルダ名は確認しやすいよう、実行日時を秒単位まで付ける。
        args.name = f"camera_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 保存先は camera_check と同じく、リポジトリ直下の runs/<name>/<name>.mp4。
    output_path = build_output_path(args.project, args.name)
    output_dir = output_path.parent
    run_logger = RunLogger(
        output_dir=output_dir,
        system_interval=args.system_log_interval,
        config={
            "arguments": vars(args),
            "source": source,
            "model": args.model,
            "time_url": TIME_URL,
            "endpoint_url": ENDPOINT_URL,
        },
    )
    # モデル読み込み中も温度やCPU負荷を残せるよう、この時点で計測を開始する。
    run_logger.start()

    # Raspberry Pi のカメラ映像を YOLO に直接渡してリアルタイム推論する。
    print(f"load model: {args.model}", flush=True)
    run_logger.log_event("INFO", f"load model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as error:
        run_logger.log_event("ERROR", repr(error))
        run_logger.stop()
        raise

    # 送信する場合だけサーバー時刻との差分を測る。
    # --no-sendではネットワークへ一切接続せず、ローカル時刻をそのまま使う。
    clock = ClockSynchronizer(
        time_url=TIME_URL,
        samples=CLOCK_SYNC_SAMPLES,
        timeout=SEND_TIMEOUT,
    )
    clock_offset = 0.0 if args.no_send else clock.synchronize()
    run_logger.log_event(
        "INFO",
        f"clock synchronized: offset_ms={round(clock_offset * 1000, 3)}",
    )

    gps_reader = GpsReader()
    payload_builder = DetectionPayloadBuilder(clock)

    sender = None

    if not args.no_send:
        sender = BackgroundResultSender(
            endpoint_url=ENDPOINT_URL,
            queue_size=QUEUE_SIZE,
            timeout=SEND_TIMEOUT,
            run_logger=run_logger,
        )
        # 送信は別スレッドで行い、通信待ちで推論ループが止まらないようにする。
        sender.start()
    else:
        run_logger.log_event("INFO", "result sending disabled")

    cap = None
    writer = None

    try:
        print(f"open source: {source}", flush=True)
        run_logger.log_event("INFO", f"open source: {source}")
        # camera_check と同じく、裏スレッドでカメラを読み続けて最新フレームだけ使う。
        cap = LatestFrameCapture(source, args.width, args.height, args.fps)
        cap.start()
        wait_for_first_frame(cap, source)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = get_capture_fourcc(cap)
        print(f"capture size: {frame_width}x{frame_height}", flush=True)
        print(
            f"capture format: {actual_fourcc} {actual_fps:.3f}fps",
            flush=True,
        )
        run_logger.log_event("INFO", f"capture size: {frame_width}x{frame_height}")
        run_logger.update_config(
            {
                "actual_capture_width": frame_width,
                "actual_capture_height": frame_height,
                "actual_capture_fps": actual_fps,
                "actual_capture_fourcc": actual_fourcc,
                "clock_offset_ms": round(clock_offset * 1000, 3),
            }
        )

        if not args.no_save:
            # YOLO の save=True ではなく OpenCV で .mp4 として保存する。
            writer = create_writer(output_path, args.output_fps, (frame_width, frame_height))
            print(f"save: {output_path}", flush=True)
            run_logger.log_event("INFO", f"save video: {output_path}")

        frame_index = 0
        camera_frame_sequence = 0

        while True:
            ok, frame, capture_timestamp_ms, next_sequence = cap.read_after(
                camera_frame_sequence
            )

            if not ok:
                if cap.running:
                    continue

                break

            camera_frame_sequence = next_sequence
            frame_index += 1
            # 実験後に処理遅延を分解できるよう、各処理の前後を個別に計測する。
            frame_started = time.perf_counter()
            inference_start_ms = epoch_ms()
            # 撮影解像度とは別に、YOLOへの入力サイズを--imgszで固定する。
            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                save=False,
                verbose=False,
            )[0]
            inference_end_ms = epoch_ms()
            render_started = time.perf_counter()
            annotated_frame = result.plot()
            render_ms = (time.perf_counter() - render_started) * 1000

            # 各フレームの推論結果に、その時点の GPS と補正済み時刻を付けて送信する。
            gps_data = gps_reader.get_current()
            payload = payload_builder.build(result=result, gps_data=gps_data)
            run_logger.log_detections(
                capture_timestamp_ms,
                {
                    **payload,
                    "camera_frame_sequence": camera_frame_sequence,
                },
            )
            send_enqueue_ms = 0.0

            if sender is not None:
                # send()はキュー投入だけを行うため、ここで測るのは通信時間ではない。
                # 実際のHTTP通信時間は送信スレッド側でnetwork.csvに記録する。
                send_started = time.perf_counter()
                sender.send(payload)
                send_enqueue_ms = (time.perf_counter() - send_started) * 1000

            detection_count = 0 if result.boxes is None else len(result.boxes)
            print(f"frame={frame_index} detections={detection_count}", flush=True)

            video_write_ms = 0.0
            if writer is not None:
                write_started = time.perf_counter()
                writer.write(annotated_frame)
                video_write_ms = (time.perf_counter() - write_started) * 1000

            label_write_ms = 0.0
            if not args.no_save_txt:
                # 検知確認用に YOLO 形式に近いラベル txt を保存する。
                label_started = time.perf_counter()
                save_label_txt(result, output_dir, frame_index)
                label_write_ms = (time.perf_counter() - label_started) * 1000

            crop_write_ms = 0.0
            if args.save_crop:
                # crop 保存は重めなので、必要なときだけ --save-crop で有効にする。
                crop_started = time.perf_counter()
                save_crops(result, frame, output_dir, frame_index)
                crop_write_ms = (time.perf_counter() - crop_started) * 1000

            speed = result.speed or {}
            # Ultralytics内部時間と、描画・保存・キュー投入時間を1行にまとめる。
            run_logger.log_timing(
                {
                    "frame_id": payload["frame_id"],
                    "capture_timestamp_ms": capture_timestamp_ms,
                    "inference_start_ms": inference_start_ms,
                    "inference_end_ms": inference_end_ms,
                    "inference_ms": inference_end_ms - inference_start_ms,
                    "preprocess_ms": round(speed.get("preprocess", 0.0), 3),
                    "yolo_inference_ms": round(speed.get("inference", 0.0), 3),
                    "postprocess_ms": round(speed.get("postprocess", 0.0), 3),
                    "render_ms": round(render_ms, 3),
                    "video_write_ms": round(video_write_ms, 3),
                    "label_write_ms": round(label_write_ms, 3),
                    "crop_write_ms": round(crop_write_ms, 3),
                    "send_enqueue_ms": round(send_enqueue_ms, 3),
                    "frame_total_ms": round(
                        (time.perf_counter() - frame_started) * 1000,
                        3,
                    ),
                    "detection_count": detection_count,
                }
            )

            if args.show:
                cv2.imshow("camera_runner", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

    except KeyboardInterrupt:
        print("stopping...")
        run_logger.log_event("INFO", "stopped by Ctrl+C")

    except Exception as error:
        run_logger.log_event("ERROR", repr(error))
        raise

    finally:
        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

        if args.show:
            cv2.destroyAllWindows()

        # 例外や Ctrl+C で抜ける場合も、送信用スレッドを終了させる。
        if sender is not None:
            sender.stop()

        run_logger.stop()


if __name__ == "__main__":
    main()
