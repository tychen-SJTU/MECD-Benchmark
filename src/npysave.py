import json
from tqdm import tqdm
import os
import torch
from encoder.Clip_encoder import clip
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import math


def generate_integers(index):
    a = index[1]
    b = index[0]
    n = a - b + 1
    if n >= 8:
        step = n // 8
        result = list(range(b, a + 1, step))[:8]
    else:
        result = list(range(b, a + 1))

    return result

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = "cuda" if torch.cuda.is_available() else "cpu"
causal_inference_file = 'your_path/MECD/VAR-main/captions/activitynet/train_caption_small3_updated_large.json'
with open(causal_inference_file, 'r') as f:
    causal_inference_list = json.load(f)

model, preprocess = clip.load("ViT-B/32", device=device)
for task_id, sample in tqdm(enumerate(causal_inference_list[516:])):
    count = 0
    key = [s for s in sample][0]
    ce_sample = sample[key]
    description = ce_sample['sentences']
    duration = ce_sample['duration']
    timestamps = ce_sample['timestamps']
    vid_val_root = '/mnt/sdb/dataset/Activitynet/v1-3/val'
    vid_train_root = '/mnt/sdb/dataset/Activitynet/v1-3/train'
    example_name = key
    vid_path = os.path.join(vid_val_root, "%s.mp4" % example_name)
    if not os.path.exists(vid_path):
        vid_path = os.path.join(vid_val_root, "%s.mkv" % example_name)
        if not os.path.exists(vid_path):
            vid_path = os.path.join(vid_train_root, "%s.mkv" % example_name)
            if not os.path.exists(vid_path):
                vid_path = os.path.join(vid_train_root, "%s.mp4" % example_name)
    print(vid_path)
    vr = VideoReader(vid_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    indies = [np.array([math.ceil(time[0] * num_frames / duration), math.floor(time[1] * num_frames / duration)]) for time in
              timestamps]
    all_frames = [generate_integers(index) for index in indies]

    events_frames = []
    for frames in all_frames:
        images_group = []
        for frame_index in frames:
            frame_index = max(frame_index, num_frames-1)
            frame_index = min(frame_index, 0)
            img = preprocess(Image.fromarray(vr[frame_index].asnumpy())).unsqueeze(0).to(device)
            image_features = model.encode_image(img)
            images_group.append(image_features)
        concatenated_images_group = torch.cat(images_group, dim=0)
        events_frames.append(concatenated_images_group)
    # print("okk")
    for i, tensor in enumerate(events_frames):
        rootpath = os.path.join('your_path/MECD/VAR-main/clip_feature/' + example_name)
        if not os.path.exists(rootpath):
            os.makedirs(rootpath)
        filename = os.path.join(rootpath, f'tensor_{i}.npy')
        np.save(filename, tensor.cpu().detach().numpy())
        # print(f'Saved {filename}')