# from flask import Flask, request, jsonify
# from threading import Thread
# from datetime import datetime, timezone
# app = Flask(__name__)
#
#
# @app.route('/endpoint', methods=['POST'])
# def receive_data():
#     # JSONデータを受信する
#     data = request.get_json()
#     current_time = datetime.now(timezone.utc).timestamp()
#     send_time = datetime.fromisoformat(data[-1]['timestamp']).timestamp()
#     communication_time = current_time - send_time
#
#     # current_time = datetime.now().timestamp()
#
#     # #print("Received data:", data)
#     print("communication_time:", communication_time)
#     # 受信したデータを表示する
#     #print("Received data:", data)
#
#     # 成功応答を返す
#     return jsonify({"status": "success", "message": "Data received"}), 200
#
#
# if __name__ == "__main__":
#     # 特定のポートでサーバーを実行する
#     app.run(host='0.0.0.0', port=40000)
import os

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/upload_video", methods=["POST"])
def upload_video():
    # リクエストにファイルが含まれているか確認する
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No video file in request"}), 400

    video_file = request.files["video"]

    # 保存パスを定義する
    save_path = os.path.join("uploads", video_file.filename)
    os.makedirs("uploads", exist_ok=True)

    # ファイルを保存する
    video_file.save(save_path)

    return jsonify({"status": "success", "message": "Video received", "path": save_path}), 200


if __name__ == "__main__":
    # サーバーを起動する
    app.run(host="0.0.0.0", port=40000)

if __name__ == "__main__":
    server_url = "http://192.168.0.237:40003/upload_video"
    file_path = "video/1.mp4"  # 動画ファイルのパス
    send_video(file_path, server_url)
