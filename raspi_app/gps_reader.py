"""GPS情報を推論結果へ追加するための読み取りインターフェース。"""


class GpsReader:
    """GPSの現在位置を共通形式で返す。現時点では未接続用の仮実装。"""

    def get_current(self):
        """現在位置を緯度・経度の辞書で返す。"""
        # TODO: 実機ではここを GPS モジュールからの読み取り処理に置き換える。
        # まだ測位できない環境でも payload の形を保てるよう None を返す。
        return {
            "lat": None,
            "lon": None,
        }
