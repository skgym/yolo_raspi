import argparse
import threading
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_source(value):
    if value.isdigit():
        return int(value)

    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--model", default="model/best2.pt", help="YOLO model path")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--project", default="runs", help="output root directory")
    parser.add_argument("--name", default="camera_check", help="output run directory name")
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


def open_capture(source, width, height, fps):
    cap = cv2.VideoCapture(source)

    if isinstance(source, int):
        # USB カメラや Raspberry Pi カメラに希望する撮影条件を伝える。
        # 実際の値はカメラ側の対応状況によって変わる場合がある。
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {source}")

    return cap


class LatestFrameCapture:
    """カメラ読み取りを別スレッドで続け、推論側には最新フレームだけ渡す。"""

    def __init__(self, source, width, height, fps):
        self.cap = open_capture(source, width, height, fps)
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()

            if not ok:
                self.running = False
                break

            with self.lock:
                # 推論が遅い場合も古いフレームを溜めず、常に最新フレームへ上書きする。
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None

            return True, self.frame.copy()

    def get(self, prop_id):
        return self.cap.get(prop_id)

    def release(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2.0)

        self.cap.release()


def create_writer(output_path, fps, frame_size):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # mp4v を使って OpenCV の VideoWriter から .mp4 として保存する。
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output_path}")

    return writer


def build_output_path(project, name):
    project_path = Path(project)

    if not project_path.is_absolute():
        # どのディレクトリから実行しても、このリポジトリ直下の runs/ に保存する。
        project_path = PROJECT_ROOT / project_path

    return project_path / name / f"{name}.mp4"


def save_label_txt(result, output_dir, frame_index):
    # 1 フレームにつき 1 つの txt を保存する。形式は YOLO の正規化 bbox に近い。
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_path = labels_dir / f"frame_{frame_index:06d}.txt"

    lines = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            x_center, y_center, width, height = box.xywhn[0].tolist()
            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}")

    label_path.write_text("\n".join(lines), encoding="utf-8")


def save_crops(result, frame, output_dir, frame_index):
    if result.boxes is None:
        return

    frame_height, frame_width = frame.shape[:2]

    for detection_index, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = max(0, min(frame_width - 1, int(x1)))
        y1 = max(0, min(frame_height - 1, int(y1)))
        x2 = max(0, min(frame_width, int(x2)))
        y2 = max(0, min(frame_height, int(y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        # 検出 bbox の範囲だけを切り出して、クラス名ごとのフォルダへ保存する。
        crop = frame[y1:y2, x1:x2]
        crop_dir = output_dir / "crops" / class_name
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_dir / f"frame_{frame_index:06d}_{detection_index:02d}.jpg"
        cv2.imwrite(str(crop_path), crop)


def main():
    args = parse_args()
    source = parse_source(args.source)
    output_path = build_output_path(args.project, args.name)
    output_dir = output_path.parent

    print(f"load model: {args.model}", flush=True)
    model = YOLO(args.model)
    print(f"open source: {source}", flush=True)
    cap = LatestFrameCapture(source, args.width, args.height, args.fps)
    cap.start()
    writer = None

    try:
        while True:
            ok, _ = cap.read()

            if ok:
                break

            if not cap.running:
                raise RuntimeError(f"failed to read source: {source}")

            time.sleep(0.01)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"capture size: {frame_width}x{frame_height}", flush=True)

        if not args.no_save:
            writer = create_writer(output_path, args.output_fps, (frame_width, frame_height))
            print(f"save: {output_path}", flush=True)

        frame_index = 0

        while True:
            try:
                ok, frame = cap.read()
            except KeyboardInterrupt:
                print("stopping...")
                break

            if not ok:
                break

            frame_index += 1
            try:
                print(f"predict frame={frame_index}", flush=True)
                # 撮影解像度は高く保ちつつ、推論負荷は --imgsz で調整する。
                result = model.predict(frame, imgsz=args.imgsz, save=False, verbose=False)[0]
            except KeyboardInterrupt:
                print("stopping...")
                break

            annotated_frame = result.plot()
            detection_count = 0 if result.boxes is None else len(result.boxes)
            print(f"frame={frame_index} detections={detection_count}", flush=True)

            if writer is not None:
                writer.write(annotated_frame)

            if not args.no_save_txt:
                save_label_txt(result, output_dir, frame_index)

            if args.save_crop:
                save_crops(result, frame, output_dir, frame_index)

            if args.show:
                cv2.imshow("camera_check", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

    except KeyboardInterrupt:
        print("stopping...")

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
