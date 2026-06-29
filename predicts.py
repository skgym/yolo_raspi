from ultralytics import YOLO

# YOLOモデルを読み込む
model = YOLO("yolov11n.pt")

# 画像に対して推論
results = model("video/test.png")

# 結果を保存・送信など、results.pyで定義されたメソッドを使える
for r in results:
    print(r.verbose())  # 検出情報を出力
    r.save_crop("output/")  # 切り出しと座標保存＋HTTP送信
