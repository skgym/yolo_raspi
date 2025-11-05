import argparse
import asyncio
import base64
import os
import copy
import aiohttp
import requests
from llava.conversation import conv_qwen
import time

start_time = time.time()

def log_time(r, *args, **kwargs):
    print(f"Response received in: {time.perf_counter() - r.elapsed.total_seconds():.6f} seconds")

def send_request(url, data):
    """同步发送POST请求"""
    try:
        t1=time.time()
        response = requests.post(url, json=data, hooks={"response": log_time})
        t2 = time.time()
        response.raise_for_status()
        t3 = time.time()

        print(t3-t2)
        print(t2 - t1)
        return response.json()
    except Exception as e:
        print(f"Error sending request to {url} with data {data}: {e}")
        return {}

def encode_image_to_base64(image_path):
    """Read an image file and encode it to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(url, image_path):
    time0=time.time()
    prompt = "<image>\nPlease generate caption towards this image."
    conv_template = copy.deepcopy(conv_qwen)
    conv_template.append_message(role=conv_template.roles[0], message=prompt)
    conv_template.append_message(role=conv_template.roles[1], message=None)
    prompt_with_template = conv_template.get_prompt()

    # Encode the local image to base64
    base64_image = encode_image_to_base64(image_path)
    time1=time.time()
    response = send_request(
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
        }
    )
    time2 = time.time()
    # print(time1-time0)
    # print(time2 - time1)
    return image_path, response['text']  # Return the filename and the response text


def test_concurrent(args):

    url = f"{args.host}:{args.port}"
    image_paths = [os.path.join(args.image_dir, file) for file in os.listdir(args.image_dir) if
                   file.lower().endswith(('.png', '.jpg', '.jpeg'))]


    # Create tasks for each image
    for image_path in image_paths:
        process_image(url, image_path)
    # tasks = [process_image(url, image_path) for image_path in image_paths]
    # results = asyncio.gather(*tasks)


    # for filename, text in results:
    #     print(f"Filename: {filename} - Caption: {text}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--image_dir", type=str, default='/home/dubx/Projects/LLaVA-NeXT/images/imgs')  # Directory containing images
    args = parser.parse_args()
    test_concurrent(args)
