import json
import logging
import os
from openai import OpenAI
import re
import openai


def extract_first_bracket_content(s):
    match = re.search(r'\[(0|1).*?\]', s)
    if match:
        return match.group(0)
    else:
        return None


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
logging.basicConfig(level=logging.INFO)

openai.api_key = os.getenv("OPENAI_API_KEY")

json_file_path = './caption/val.json' # Replace with your file path
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
        if 'sentences' in value:
            lengths = [len(value['sentences']) - 1 - i for i in range(len(value['sentences']) - 1)]
            prompts[key] = (task + '\nText description:\n' +
                            str(value['sentences']) + '\n\n' +
                            'Your probability output lists(length = [{}]) are:\n'.format(','.join(map(str, lengths))) +
                            'Please do not generate any explanation.')


# 遍历prompts
with open('gpt4_whole3.txt', 'w') as file:
    for i, (video_id, prompt) in enumerate(prompts.items()):
        logging.info(f"Processing {video_id}")
        attempts = 0
        success = False
        while attempts < 3 and not success:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="gpt-4",
                )
                # 确保正确访问chat_completion对象
                output = chat_completion.choices[0].message.content
                if '[[' in output:
                    pred = ('[[' + output.split('[[')[1]).split(']]')[0] + ']]'
                elif '[ [' in output:
                    pred = ('[[' + output.split('[ [')[1]).split('] ]')[0] + ']]'
                else:
                    pred = ('[' + output.split('[')[1]).split(']')[0] + ']'
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