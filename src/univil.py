import torch
from encoder import clip
from PIL import Image
import json
import numpy as np
from decord import VideoReader, cpu
import os
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn


def freeze_parameters(m, requires_grad=False):
    if m is None:
        return

    if isinstance(m, nn.Parameter):
        m.requires_grad = requires_grad
    else:
        for p in m.parameters():
            p.requires_grad = requires_grad

    return m


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


class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)  # 使用 Sigmoid 激活函数输出二分类结果
        return x[0, :]


num_epochs = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-L/14", device=device)
model.eval()
model = freeze_parameters(model)
classifier = BinaryClassificationModel(input_size=768).to(device)
classifier.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

causal_inference_file = 'your_path/MECD/VAR-main/captions/activitynet/train_caption_small3_updated_large.json'
with open(causal_inference_file, 'r') as f:
    causal_inference_list = json.load(f)

val_causal_inference_file = 'your_path/MECD/VAR-main/captions/activitynet/val_caption_small3_updated_large.json'
with open(val_causal_inference_file, 'r') as f:
    val_causal_inference_list = json.load(f)

from torch.utils.data import Dataset, DataLoader


class MECDDataset(Dataset):
    def __init__(self, causal_inference_list):
        self.causal_inference_list = causal_inference_list
    def __len__(self):
        return len(self.causal_inference_list)
    def __getitem__(self, task_id):
        sample = self.causal_inference_list[task_id]
        key = [s for s in sample][0]
        ce_sample = sample[key]
        timestamps = ce_sample['timestamps']
        duration = ce_sample['duration']

        vid_val_root = '/mnt/sdb/dataset/Activitynet/v1-3/val'
        vid_train_root = '/mnt/sdb/dataset/Activitynet/v1-3/train'
        vid_path = os.path.join(vid_val_root, "%s.mp4" % key)
        if not os.path.exists(vid_path):
            vid_path = os.path.join(vid_val_root, "%s.mkv" % key)
            if not os.path.exists(vid_path):
                vid_path = os.path.join(vid_train_root, "%s.mkv" % key)
                if not os.path.exists(vid_path):
                    vid_path = os.path.join(vid_train_root, "%s.mp4" % key)

        vr = VideoReader(vid_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        indies = [np.array([int(time[0] * num_frames / duration), int(time[1] * num_frames / duration)]) for time in
                  timestamps]
        all_frames = [generate_integers(index) for index in indies]

        events_vision_feature = []
        for frames in all_frames:
            images_group = []
            for frame_index in frames:
                if abs(frame_index-len(all_frames)) < 1:
                    frame_index = frame_index-1
                img = preprocess(Image.fromarray(vr[frame_index].asnumpy())).unsqueeze(0)
                images_group.append(img)
            concatenated_images_group = torch.cat(images_group, dim=0)
            events_vision_feature.append(concatenated_images_group)
        return task_id, sample, all_frames, events_vision_feature

train_dataloader = DataLoader(MECDDataset(causal_inference_list), num_workers=32, collate_fn=lambda x: x[0])
val_dataloader = DataLoader(MECDDataset(val_causal_inference_list), num_workers=32, collate_fn=lambda x: x[0])
for epoch in range(num_epochs):
    epoch_loss = 0
    correct_predictions = 0
    total_samples = 0
    for task_id, sample, all_frames, events_vision_feature in tqdm(train_dataloader):
        key = [s for s in sample][0]
        ce_sample = sample[key]
        descriptions = ce_sample['sentences']
        relation = ce_sample['relation']
        relation = relation[0]
        relation = [int(char) for char in relation]
        events_vision_feature = [model.encode_image(f.to(device)) for f in events_vision_feature]
        events_text_feature = []
        events_multimodal_feature = []
        for i, sentence in enumerate(descriptions):
            text = clip.tokenize(sentence).to(device)
            text_features = model.encode_text(text)
            multimodal_feature = torch.cat((text_features, events_vision_feature[i]), dim=0)
            events_text_feature.append(text_features)
            del text_features
            events_multimodal_feature.append(multimodal_feature)
            del multimodal_feature
        output_list = []
        for event in events_multimodal_feature[:-1]:
            events_pair = torch.cat((event, events_multimodal_feature[-1]), dim=0)
            output = classifier(events_pair.to(torch.float32))
            output_list.append(output)
            del output
        tensor_stack = torch.stack(output_list)
        labels = torch.tensor(relation).to(device)
        tensor_stack = torch.cat((1 - tensor_stack, tensor_stack), dim=1)
        loss = criterion(tensor_stack, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss
        torch.cuda.empty_cache()
        if task_id % 20 == 0 and task_id > 0:
            print(epoch_loss / (task_id + 1))
            print(correct_predictions / total_samples)
        _, predicted = torch.max(tensor_stack, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        del tensor_stack, predicted, labels
        torch.cuda.empty_cache()
    epoch_acc = correct_predictions / total_samples
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    print("eval")
    eval_correct_prediction = 0
    eval_samples = 0
    with torch.no_grad():
        for task_id, sample, all_frames, events_vision_feature in tqdm(val_dataloader):
            key = [s for s in sample][0]
            ce_sample = sample[key]
            descriptions = ce_sample['sentences']
            relation = ce_sample['relation']
            relation = relation[0]
            relation = [int(char) for char in relation]

            events_vision_feature = [model.encode_image(f.to(device)) for f in events_vision_feature]
            events_text_feature = []
            events_multimodal_feature = []
            for i, sentence in enumerate(descriptions):
                text = clip.tokenize(sentence).to(device)
                text_features = model.encode_text(text)
                multimodal_feature = torch.cat((text_features, events_vision_feature[i]), dim=0)
                events_text_feature.append(text_features)
                del text_features
                events_multimodal_feature.append(multimodal_feature)
                del multimodal_feature
            output_list = []
            for event in events_multimodal_feature[:-1]:
                events_pair = torch.cat((event, events_multimodal_feature[-1]), dim=0)
                output = classifier(events_pair.to(torch.float32))
                output_list.append(output)
                del output
            tensor_stack = torch.stack(output_list)  # 形状变为 (n, 1)
            labels = torch.tensor(relation).to(device)  # 形状为 (n,)
            tensor_stack = torch.cat((1 - tensor_stack, tensor_stack), dim=1)  # 形状变为 (n, 2)

            _, predicted = torch.max(tensor_stack, 1)
            eval_correct_prediction += (predicted == labels).sum().item()
            eval_samples += labels.size(0)
        eval_epoch_acc = eval_correct_prediction / eval_samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Eval Accuracy: {eval_epoch_acc:.4f}")