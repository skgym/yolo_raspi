import os
import argparse
import asyncio
import base64
import aiohttp
from PIL import Image
from io import BytesIO
from ultralytics import YOLO


async def send_request(url, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as resp:
            return await resp.json()


def encode_image_to_base64(image):
    """Convert a PIL image to base64 encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


async def process_and_send_images(url, image_paths, model):
    responses = []
    for image_path in image_paths:
        # Run YOLO model prediction
        results = model.predict(image_path, save=False, imgsz=640, conf=0.15, vid_stride=1, save_txt=False)
        for *xyxy, conf, cls in results.xyxy[0]:  # Assuming xyxy format for bounding boxes
            # Crop detected objects
            img = Image.open(image_path)
            crop = img.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
            base64_image = encode_image_to_base64(crop)

            # Prepare the data for sending
            data = {
                "image_data": f"data:image/jpeg;base64,{base64_image}"
            }
            response = await send_request(url, data)
            responses.append(response)
    return responses


async def main(args):
    model = YOLO(args.weights_path)
    image_paths = [os.path.join(args.image_dir, file) for file in os.listdir(args.image_dir) if
                   file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = await process_and_send_images(f"http://{args.host}:{args.port}", image_paths, model)
    for result in results:
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--image_dir", type=str, default='/home/dubx/Projects/LLaVA-NeXT/images/imgs')
    parser.add_argument("--weights_path", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(main(args))
