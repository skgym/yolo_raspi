"""カメラ映像でYOLOの検知精度と保存結果だけを確認するためのプログラム。"""

import argparse
import threading
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_source(value):
    """数字だけの入力はカメラ番号、それ以外は動画やストリームのパスとして扱う。"""
    if value.isdigit():
        return int(value)

    return value


def parse_args():
    """コマンドラインからカメラ、推論、保存に関する設定を受け取る。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--model", default="model/best2.pt", help="YOLO model path")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--project", default="runs", help="output root directory")
    parser.add_argument("--name", default="camera_check", help="output run directory name")
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
    parser.add_argument("--no-save", action="store_true", help="do not save annotated mp4")
    parser.add_argument("--no-save-txt", action="store_true", help="do not save label txt files")
    parser.add_argument("--save-crop", action="store_true", help="save cropped detections")
    return parser.parse_args()


def open_capture(source, width, height, fps):
    """OpenCVで入力を開き、カメラの場合は希望する撮影条件を設定する。"""
    cap = cv2.VideoCapture(source)

    if isinstance(source, int):
        # USB カメラや Raspberry Pi カメラに希望する撮影条件を伝える。
        # 実際の値はカメラ側の対応状況によって変わる場合がある。
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {source}")

    return cap


class LatestFrameCapture:
    """カメラ読み取りを別スレッドで続け、推論側には最新フレームだけ渡す。"""

    def __init__(self, source, width, height, fps):
        """入力を開き、最新フレーム共有用の状態を初期化する。"""
        self.cap = open_capture(source, width, height, fps)
        self.frame = None
        self.captured_at_ms = None
        self.frame_sequence = 0
        self.running = False
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.thread = None

    def start(self):
        """カメラを読み続けるバックグラウンドスレッドを開始する。"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        """入力が終了するまで撮影し、共有フレームを最新版へ更新する。"""
        while self.running:
            ok, frame = self.cap.read()

            if not ok:
                self.running = False
                break

            with self.condition:
                # 推論が遅い場合も古いフレームを溜めず、常に最新フレームへ上書きする。
                self.frame = frame
                self.captured_at_ms = time.time_ns() // 1_000_000
                self.frame_sequence += 1
                self.condition.notify_all()

    def read(self):
        """推論中に上書きされないよう、最新フレームのコピーを返す。"""
        with self.lock:
            if self.frame is None:
                return False, None

            return True, self.frame.copy()

    def read_with_metadata(self):
        """最新フレームと、撮影時刻・カメラ側の連番を同時に取得する。"""
        with self.lock:
            if self.frame is None:
                return False, None, None, None

            return True, self.frame.copy(), self.captured_at_ms, self.frame_sequence

    def read_after(self, frame_sequence, timeout=0.5):
        """指定した連番より新しいフレームを待ち、同じ画像の重複推論を防ぐ。"""
        with self.condition:
            self.condition.wait_for(
                lambda: self.frame_sequence > frame_sequence or not self.running,
                timeout=timeout,
            )

            if self.frame is None or self.frame_sequence <= frame_sequence:
                return False, None, None, frame_sequence

            return True, self.frame.copy(), self.captured_at_ms, self.frame_sequence

    def get(self, prop_id):
        """OpenCVのカメラ設定値を取得する。"""
        return self.cap.get(prop_id)

    def release(self):
        """読み取りスレッドを停止してカメラを解放する。"""
        with self.condition:
            self.running = False
            self.condition.notify_all()

        if self.thread is not None:
            self.thread.join(timeout=2.0)

        self.cap.release()


def create_writer(output_path, fps, frame_size):
    """指定した解像度とFPSでMP4を書き込むVideoWriterを作る。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # mp4v を使って OpenCV の VideoWriter から .mp4 として保存する。
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output_path}")

    return writer


def get_capture_fourcc(cap):
    """カメラが実際に採用したFOURCCをMJPGなどの文字列へ変換する。"""
    value = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))


def build_output_path(project, name):
    """実行場所に左右されない絶対パスの保存先を組み立てる。"""
    project_path = Path(project)

    if not project_path.is_absolute():
        # どのディレクトリから実行しても、このリポジトリ直下の runs/ に保存する。
        project_path = PROJECT_ROOT / project_path

    return project_path / name / f"{name}.mp4"


def save_label_txt(result, output_dir, frame_index):
    """検知枠をYOLO形式に近いテキストとしてフレームごとに保存する。"""
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
    """検知枠の内側だけを切り出し、クラス別フォルダへ画像保存する。"""
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
    """送信処理を行わず、カメラ入力の推論と確認用データ保存を実行する。"""
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
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = get_capture_fourcc(cap)
        print(f"capture size: {frame_width}x{frame_height}", flush=True)
        print(
            f"capture format: {actual_fourcc} {actual_fps:.3f}fps",
            flush=True,
        )

        if not args.no_save:
            writer = create_writer(output_path, args.output_fps, (frame_width, frame_height))
            print(f"save: {output_path}", flush=True)

        frame_index = 0
        camera_frame_sequence = 0

        while True:
            try:
                ok, frame, _, next_sequence = cap.read_after(camera_frame_sequence)
            except KeyboardInterrupt:
                print("stopping...")
                break

            if not ok:
                if cap.running:
                    continue

                break

            camera_frame_sequence = next_sequence
            frame_index += 1
            try:
                print(f"predict frame={frame_index}", flush=True)
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
