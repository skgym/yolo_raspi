"""実験条件、検知結果、処理時間、通信状態、端末負荷を低負荷で保存する。"""

import csv
import json
import os
import platform
import threading
import time
from pathlib import Path


def epoch_ms():
    """現在時刻を Unix epoch のミリ秒で返す。"""
    return time.time_ns() // 1_000_000


class RunLogger:
    """実験の再現と性能評価に必要な情報を、低負荷な形式で保存する。"""

    # 1フレームの処理を推論、描画、保存、送信キュー投入に分けて記録する列。
    # *_timestamp_ms はUnix時刻、*_msは処理にかかった時間を表す。
    TIMING_FIELDS = (
        "frame_id",
        "capture_timestamp_ms",
        "inference_start_ms",
        "inference_end_ms",
        "inference_ms",
        "preprocess_ms",
        "yolo_inference_ms",
        "postprocess_ms",
        "render_ms",
        "video_write_ms",
        "label_write_ms",
        "crop_write_ms",
        "send_enqueue_ms",
        "frame_total_ms",
        "detection_count",
    )
    # 送信成功・失敗・キューからの破棄を同じ形式で比較するための列。
    NETWORK_FIELDS = (
        "timestamp_ms",
        "frame_id",
        "event",
        "status_code",
        "elapsed_ms",
        "payload_items",
        "error",
    )
    # Raspberry Piの性能低下や熱によるクロック低下を確認するための列。
    SYSTEM_FIELDS = (
        "timestamp_ms",
        "cpu_percent",
        "load_1m",
        "memory_used_percent",
        "memory_available_mb",
        "process_rss_mb",
        "process_cpu_percent",
        "cpu_temp_c",
        "cpu_freq_mhz",
        "disk_free_mb",
        "network_rx_bytes",
        "network_tx_bytes",
    )

    def __init__(self, output_dir, config, system_interval=1.0):
        """保存先と実験条件を設定し、各ログファイルを開く。"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started_at_ms = epoch_ms()
        # 短すぎる間隔は計測自体が負荷になるため、最小200msに制限する。
        self.system_interval = max(0.2, float(system_interval))
        # 推論スレッド、送信スレッド、負荷計測スレッドの同時書き込みを保護する。
        self.lock = threading.Lock()
        self.running = False
        self.system_thread = None
        # 終了時のrun_summary.jsonへ書く集計値。
        self.frame_count = 0
        self.detection_count = 0
        self.send_success_count = 0
        self.send_failure_count = 0
        self.dropped_payload_count = 0

        # コマンド引数だけでなく、実行OSとPythonバージョンも再現用に保存する。
        self.runtime_config = {
            "started_at_ms": self.started_at_ms,
            "time_unit": "milliseconds",
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pid": os.getpid(),
            **config,
        }
        self._write_json(self.output_dir / "run_config.json", self.runtime_config)

        # JSONLは1行が1レコードなので、途中停止しても読み取れる範囲が残る。
        # 64KBバッファにまとめて書き、フレームごとのディスクI/Oを減らす。
        self.detections_file = (self.output_dir / "detections.jsonl").open(
            "a", encoding="utf-8", buffering=65536
        )
        self.sent_payloads_file = (self.output_dir / "sent_payloads.jsonl").open(
            "a", encoding="utf-8", buffering=65536
        )
        self.run_log_file = (self.output_dir / "run.log").open(
            "a", encoding="utf-8", buffering=65536
        )
        self.timing_file, self.timing_writer = self._open_csv(
            "timing.csv", self.TIMING_FIELDS
        )
        self.network_file, self.network_writer = self._open_csv(
            "network.csv", self.NETWORK_FIELDS
        )
        self.system_file, self.system_writer = self._open_csv(
            "system_metrics.csv", self.SYSTEM_FIELDS
        )
        self._previous_cpu = None
        self._previous_process_cpu = None

    def _open_csv(self, name, fields):
        """CSVを追記モードで開き、新規ファイルの場合だけ見出しを書く。"""
        path = self.output_dir / name
        is_empty = not path.exists() or path.stat().st_size == 0
        file_obj = path.open("a", encoding="utf-8", newline="", buffering=65536)
        writer = csv.DictWriter(file_obj, fieldnames=fields)

        if is_empty:
            writer.writeheader()

        return file_obj, writer

    @staticmethod
    def _write_json(path, value):
        """設定や集計の小さな辞書を、人が読みやすいJSONで保存する。"""
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def update_config(self, values):
        """カメラを開いた後に判明する実測値を設定ファイルへ追記する。"""
        with self.lock:
            self.runtime_config.update(values)
            self._write_json(
                self.output_dir / "run_config.json",
                self.runtime_config,
            )

    def start(self):
        """システム負荷を一定間隔で測るバックグラウンド処理を開始する。"""
        self.running = True
        self.log_event("INFO", "run started")
        self.system_thread = threading.Thread(
            target=self._system_worker,
            name="system-metrics",
            daemon=True,
        )
        self.system_thread.start()

    def log_event(self, level, message):
        """起動、停止、例外などのイベントをミリ秒時刻付きで記録する。"""
        record = {
            "timestamp_ms": epoch_ms(),
            "level": level,
            "message": str(message),
        }

        with self.lock:
            self.run_log_file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    def log_detections(self, capture_timestamp_ms, payload):
        """1フレーム分の撮影時刻、GPS、全検知結果をJSONLへ記録する。"""
        record = {
            "recorded_at_ms": epoch_ms(),
            "capture_timestamp_ms": capture_timestamp_ms,
            **payload,
        }

        with self.lock:
            self.detections_file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            self.frame_count += 1
            self.detection_count += len(payload.get("detections", []))

    def log_timing(self, values):
        """1フレーム内の各処理時間をtiming.csvへ記録する。"""
        row = {field: values.get(field, "") for field in self.TIMING_FIELDS}

        with self.lock:
            self.timing_writer.writerow(row)

    def log_network(
        self,
        *,
        frame_id,
        event,
        status_code="",
        elapsed_ms="",
        payload_items=0,
        error="",
        payload=None,
    ):
        """送信結果を記録し、送信を試みたJSONも必要に応じて保存する。"""
        timestamp_ms = epoch_ms()
        row = {
            "timestamp_ms": timestamp_ms,
            "frame_id": frame_id,
            "event": event,
            "status_code": status_code,
            "elapsed_ms": elapsed_ms,
            "payload_items": payload_items,
            "error": error,
        }

        with self.lock:
            self.network_writer.writerow(row)

            # 実行終了時に送信品質をすぐ確認できるよう、種類別に集計する。
            if event == "sent":
                self.send_success_count += 1
            elif event == "failed":
                self.send_failure_count += 1
            elif event == "dropped":
                self.dropped_payload_count += 1

            # droppedはHTTP送信していないため、network.csvだけに記録する。
            if payload is not None:
                sent_record = {
                    "recorded_at_ms": timestamp_ms,
                    "frame_id": frame_id,
                    "event": event,
                    "status_code": status_code,
                    "elapsed_ms": elapsed_ms,
                    "error": error,
                    "payload": payload,
                }
                self.sent_payloads_file.write(
                    json.dumps(
                        sent_record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    def _system_worker(self):
        """指定間隔で端末状態を読み、まとめてファイルへ反映する。"""
        while self.running:
            started = time.monotonic()

            try:
                metrics = self._read_system_metrics()

                with self.lock:
                    self.system_writer.writerow(metrics)
                    # 最大でも計測間隔分だけログを失う設計にしつつI/O回数を抑える。
                    self._flush_files()
            except (OSError, ValueError, KeyError) as error:
                self.log_event("WARNING", f"system metrics failed: {error}")

            remaining = self.system_interval - (time.monotonic() - started)

            if remaining > 0:
                time.sleep(remaining)

    def _read_system_metrics(self):
        """1回分のCPU、メモリ、温度、ストレージ、通信量を取得する。"""
        cpu_percent = self._read_cpu_percent()
        memory = self._read_memory()
        rx_bytes, tx_bytes = self._read_network_bytes()

        try:
            load_1m = round(os.getloadavg()[0], 3)
        except OSError:
            load_1m = ""

        return {
            "timestamp_ms": epoch_ms(),
            "cpu_percent": cpu_percent,
            "load_1m": load_1m,
            "memory_used_percent": memory["used_percent"],
            "memory_available_mb": memory["available_mb"],
            "process_rss_mb": self._read_process_rss_mb(),
            "process_cpu_percent": self._read_process_cpu_percent(),
            "cpu_temp_c": self._read_cpu_temp(),
            "cpu_freq_mhz": self._read_cpu_freq(),
            "disk_free_mb": self._read_disk_free_mb(),
            "network_rx_bytes": rx_bytes,
            "network_tx_bytes": tx_bytes,
        }

    def _read_cpu_percent(self):
        """Linuxの/proc/statの差分から端末全体のCPU使用率を計算する。"""
        values = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        ticks = [int(value) for value in values]
        idle = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)
        total = sum(ticks)
        current = (idle, total)

        if self._previous_cpu is None:
            self._previous_cpu = current
            return ""

        previous_idle, previous_total = self._previous_cpu
        self._previous_cpu = current
        total_delta = total - previous_total

        if total_delta <= 0:
            return ""

        idle_delta = idle - previous_idle
        return round(100.0 * (1.0 - idle_delta / total_delta), 2)

    @staticmethod
    def _read_memory():
        """Linuxの/proc/meminfoからメモリ使用率と空き容量を取得する。"""
        values = {}

        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])

        total_kb = values["MemTotal"]
        available_kb = values["MemAvailable"]
        used_percent = 100.0 * (total_kb - available_kb) / total_kb
        return {
            "used_percent": round(used_percent, 2),
            "available_mb": round(available_kb / 1024.0, 2),
        }

    @staticmethod
    def _read_process_rss_mb():
        """このYOLOプロセスが物理メモリを何MB使っているか取得する。"""
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024.0, 2)

        return ""

    def _read_process_cpu_percent(self):
        """このYOLOプロセスが使用したCPU時間の差分から使用率を計算する。"""
        columns = Path("/proc/self/stat").read_text(encoding="utf-8").split()
        process_ticks = int(columns[13]) + int(columns[14])
        current = (process_ticks, time.monotonic())

        if self._previous_process_cpu is None:
            self._previous_process_cpu = current
            return ""

        previous_ticks, previous_time = self._previous_process_cpu
        self._previous_process_cpu = current
        elapsed = current[1] - previous_time

        if elapsed <= 0:
            return ""

        cpu_seconds = (process_ticks - previous_ticks) / os.sysconf("SC_CLK_TCK")
        return round(100.0 * cpu_seconds / elapsed, 2)

    @staticmethod
    def _read_cpu_temp():
        """Raspberry PiのCPU温度を取得し、取得できない環境では空欄にする。"""
        path = Path("/sys/class/thermal/thermal_zone0/temp")

        if not path.exists():
            return ""

        return round(float(path.read_text(encoding="utf-8").strip()) / 1000.0, 2)

    @staticmethod
    def _read_cpu_freq():
        """熱制限などによる性能低下を確認するため、現在のCPU周波数を取得する。"""
        path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")

        if not path.exists():
            return ""

        return round(float(path.read_text(encoding="utf-8").strip()) / 1000.0, 2)

    def _read_disk_free_mb(self):
        """長時間実験中の容量不足を検知できるよう、保存先の空き容量を取得する。"""
        stats = os.statvfs(self.output_dir)
        return round(stats.f_bavail * stats.f_frsize / (1024.0 * 1024.0), 2)

    @staticmethod
    def _read_network_bytes():
        """全ネットワークインターフェースの累積送受信バイト数を取得する。"""
        rx_total = 0
        tx_total = 0

        for line in Path("/proc/net/dev").read_text(encoding="utf-8").splitlines()[2:]:
            _, values = line.split(":", 1)
            columns = values.split()
            rx_total += int(columns[0])
            tx_total += int(columns[8])

        return rx_total, tx_total

    def _flush_files(self):
        """メモリ上のログを全ファイルへまとめて書き出す。"""
        for file_obj in (
            self.detections_file,
            self.sent_payloads_file,
            self.run_log_file,
            self.timing_file,
            self.network_file,
            self.system_file,
        ):
            file_obj.flush()

    def stop(self):
        """負荷計測を止め、実験全体の集計を書いて全ファイルを閉じる。"""
        if not self.running:
            return

        self.running = False

        if self.system_thread is not None:
            self.system_thread.join(timeout=self.system_interval + 0.5)

        ended_at_ms = epoch_ms()
        # 実験の概要をCSV集計なしでも確認できるよう、主要件数をJSONへまとめる。
        summary = {
            "started_at_ms": self.started_at_ms,
            "ended_at_ms": ended_at_ms,
            "duration_ms": ended_at_ms - self.started_at_ms,
            "processed_frames": self.frame_count,
            "detections": self.detection_count,
            "send_successes": self.send_success_count,
            "send_failures": self.send_failure_count,
            "dropped_payloads": self.dropped_payload_count,
        }
        self._write_json(self.output_dir / "run_summary.json", summary)
        self.log_event("INFO", "run stopped")

        for file_obj in (
            self.detections_file,
            self.sent_payloads_file,
            self.run_log_file,
            self.timing_file,
            self.network_file,
            self.system_file,
        ):
            file_obj.close()
