class GpsReader:
    def get_current(self):
        # TODO: 実機ではここを GPS モジュールからの読み取り処理に置き換える。
        # まだ測位できない環境でも payload の形を保てるよう None を返す。
        return {
            "lat": None,
            "lon": None,
        }
