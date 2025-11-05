import asyncio
import base64
import io
import imagehash
from PIL import Image


def get_phash(data):
    if isinstance(data, str):
        # 如果符合文件路径的形式，可以尝试 open
        # 或者如果您知道它是 base64，可以尝试 base64 解码
        # 这里示例一下 base64 解码:
        data = base64.b64decode(data)
        # 到这一步，data 应该是 bytes
    pil_image = Image.open(io.BytesIO(data))


    return str(imagehash.phash(pil_image))


def hamming_distance(hash1, hash2):
    """
    比较两个 pHash 字符串的海明距离。
    若距离小于一定阈值，则认为两张图像近似或相同。
    """
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))