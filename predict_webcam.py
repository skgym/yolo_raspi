import cv2

# ウェブカメラを開く（デバイスインデックスは 0、複数のカメラがある場合は 1、2 などのインデックスを試してください）
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# ウェブカメラが正常に開いたか確認する
if not cap.isOpened():
    print("ウェブカメラを開けません")
    exit()

# 1フレームを読み取る
ret, frame = cap.read()
if ret:
    # 画像をローカルに保存する
    cv2.imwrite("captured_image.jpg", frame)
    print("画像は 'captured_image.jpg' として保存されました")
else:
    print("画像をキャプチャできません")

# ウェブカメラのリソースを解放する
cap.release()