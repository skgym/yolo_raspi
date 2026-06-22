import asyncio
import json

import aiohttp


async def send_data(detection_results):
    url = "http://192.168.0.122:40000/endpoint"  # Replace with the actual IP address and port of the Isaac Sim machine   http://192.168.0.237:40000/endpoint
    headers = {"Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, data=json.dumps(detection_results), headers=headers) as response:
                resp_text = await response.text()
                print(f"Server response: {resp_text}")
        except Exception as e:
            print(f"Error sending data: {e}")


async def main():
    detection_results = [{"filename": "im_1_classA_1.jpg", "class_name": "car", "bbox": [240.4, 637.2, 474.8, 761.4]}]
    await send_data(detection_results)


if __name__ == "__main__":
    asyncio.run(main())
