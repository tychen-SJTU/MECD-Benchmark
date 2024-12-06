from openai import OpenAI
import os
from PIL import Image
import base64
from io import BytesIO
import json
import logging
import re


api_key = ' '
base_url = ' '
client = OpenAI(api_key=api_key, base_url=base_url)
prompt_path = 'prompt.txt'
prompts = {}
with open(prompt_path, encoding="utf-8") as f:
    task = f.readlines()
task = ''.join(task)

json_file_path = 'captions/test.json'
with open(json_file_path, 'r', encoding="utf-8") as file:
    data = json.load(file)

all_sentences = []
for item in data:
    for key, value in item.items():
        lengths = [len(value['sentences']) - 1 - i for i in range(len(value['sentences']) - 1)]
        prompts[key] = (task +
                        'Text descriptions of {} events:\n'.format(len(value['sentences'])) +
                        str(value['sentences']) + '\n' +
                        'Images that describe {} events are also provided.'
                        .format(len(value['sentences'])) +
                        'Your probability output lists(length = [{}]) are:\n'.format(','.join(map(str, lengths))) +
                        'Please do not generate any explanation.')


def extract_first_bracket_content(s):
    match = re.search(r'\[(0|1).*?\]', s)
    if match:
        return match.group(0)
    else:
        return None


def encode_image(image):
    if isinstance(image, str):
        image = Image.open(image)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes)
    img_str = img_base64.decode('utf-8')

    return img_str


def encode_image_gpt4v(image):
    return 'data:image/jpeg;base64,' + encode_image(image)


def request_gpt4v(prompt: str, images: list, detail='auto'):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] +
                           [{"type": "image_url", "image_url": {"url": encode_image_gpt4v(image), "detail": detail}}
                            for image in images]
            }
        ],
    )
    return response.choices[0].message.content


with open('gpt4o.txt', 'w') as file:
    for i, (video_id, prompt1) in enumerate(prompts.items()):
        attempts = 0
        success = False
        image_folder = os.path.join("activitynet_image", video_id)
        image_paths = []
        for root, dirs, files in os.walk(image_folder):
            for file in files:
                image_paths.append(os.path.join(root, file))
        while attempts < 3 and not success:
            try:
                logging.info(f"Processing {video_id}, Attempt {attempts + 1}")
                output = request_gpt4v(prompt1, image_paths)
                pred = extract_first_bracket_content("".join(output))
                if pred is not None:
                    pred_list = pred.split(',')
                    if len(pred_list) > 2:
                        file.write(f"Video ID: {video_id}\nOutput: {pred}\n\n")
                        progress = ((i + 1) / len(prompts)) * 100
                        print(f"{progress:.2f}%")
                        success = True
            except Exception as e:
                attempts += 1
                logging.error(f"Error processing {video_id}, Attempt {attempts}: {e}")
                if attempts == 3:
                    logging.error(f"Failed to process {video_id} after 3 attempts.")
