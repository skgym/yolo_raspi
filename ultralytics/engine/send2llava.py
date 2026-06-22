import argparse
import asyncio
import base64
import copy
import os
import time

import aiohttp
from llava.conversation import conv_qwen


async def send_request(url, data, delay=0):
    await asyncio.sleep(delay)
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        async with session.post(url, json=data) as resp:
            output = await resp.json()
        end_time = time.time()
    return output, end_time - start_time


def encode_image_to_base64(image_path):
    """Read an image file and encode it to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def process_image(url, image_path):
    # prompt = "<image>\nPlease generate caption towards this image."
    prompt = '<image>\nPlease generate clear descriptions (color,gender,type) of the mian object in the input image in this format: "A white car." "A woman wearing a red top and blue pants."" A man in a red shirt riding a black motorcycle."'
    conv_template = copy.deepcopy(conv_qwen)
    conv_template.append_message(role=conv_template.roles[0], message=prompt)
    conv_template.append_message(role=conv_template.roles[1], message=None)
    prompt_with_template = conv_template.get_prompt()

    # Encode the local image to base64
    base64_image = encode_image_to_base64(image_path)
    response, request_time = await send_request(
        url + "/generate",
        {
            "text": prompt_with_template,
            "image_data": f"data:image/jpeg;base64,{base64_image}",
            "sampling_params": {
                "max_new_tokens": 1024,
                "temperature": 0,
                "top_p": 1.0,
                "presence_penalty": 2,
                "frequency_penalty": 2,
                "stop": "<|im_end|>",
            },
        },
    )
    return image_path, response["text"], request_time  # Return the filename and the response text


async def test_concurrent(args):
    url = f"{args.host}:{args.port}"
    image_paths = [
        os.path.join(args.image_dir, file)
        for file in os.listdir(args.image_dir)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Create tasks for each image
    tasks = [process_image(url, image_path) for image_path in image_paths]
    results = await asyncio.gather(*tasks)

    # Print each filename and its corresponding result
    for filename, text, elapsed_time in results:
        # print(f"{filename} --- {text}")
        print(f"{filename} --- {text} --- Time taken: {elapsed_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--image_dir", type=str, default="/home/dubx/Projects/yolov11/video/imgs"
    )  # Directory containing images
    args = parser.parse_args()
    asyncio.run(test_concurrent(args))


# ea81951eb17ce661

# 4.0738 seconds  4.0714
# 0.0434

# 0.7870
# 0.3997
