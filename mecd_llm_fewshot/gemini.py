import google.generativeai as genai
import pathlib
import logging
import re
import json
import os
import time
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

def extract_first_bracket_content(s):
    match = re.search(r'\[(0|1).*?\]', s)
    if match:
        return match.group(0)
    else:
        return None

genai.configure(api_key=GOOGLE_API_KEY, transport='rest')
# models/gemini-1.0-pro
# models/gemini-1.0-pro-001
# models/gemini-1.0-pro-latest
# models/gemini-1.0-pro-vision-latest
# models/gemini-1.5-flash
# models/gemini-1.5-flash-001
# models/gemini-1.5-flash-latest
# models/gemini-1.5-pro
# models/gemini-1.5-pro-001
# models/gemini-1.5-pro-latest
# models/gemini-pro
# models/gemini-pro-vision
model = genai.GenerativeModel('gemini-1.5-pro')

# picture = {
#     'mime_type': 'image/png',
#     'data': pathlib.Path('activitynet_image/v_-fMxoShIXiM/0.jpg').read_bytes()
# }
json_file_path = './caption/test.json'  # Replace with your file path
prompts = {}

prompt_path = 'prompt.txt'
with open(prompt_path, encoding="utf-8") as f:
    task = f.readlines()
task = ''.join(task)

remove_timeline = True
with open(json_file_path, 'r', encoding="utf-8") as file:
    data = json.load(file)

all_sentences = []
for item in data:
    for key, value in item.items():
        prompts[key] = (task +
                        'Textual descriptions of {} events:\n'.format(len(value['sentences'])) +
                        str(value['sentences']) + '\n' +
                        'Images that describe {} events are also provided.'
                        .format(len(value['sentences'])) +
                        'You should determine causality mainly through textual descriptions. ' +
                        'Your probability output is:\n')


with open('gemini.txt', 'w') as file:
    for i, (video_id, prompt1) in enumerate(prompts.items()):
        # time.sleep(1)
        attempts = 0
        success = False
        image_folder = pathlib.Path(os.path.join("0set", video_id))
        images_data = pathlib.Path(os.path.join(image_folder, "a.jpg")).read_bytes()

        # for image in image_paths:
        images = {
            'mime_type': 'image/jpg',
            'data': images_data
        }
        while attempts < 3 and not success:
            try:
                logging.info(f"Processing {video_id}, Attempt {attempts + 1}")
                output = model.generate_content(contents=[prompt1, images]).text
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
