import json
import pandas

# Load dict from json file


options_to_idx = {'A': 1, 'B':2, 'C':3, 'D':4, 'E':5}

results_path = '/home/hbliu/Ask-Anything/video_chat2/results/whole_test/videochat2_llama.json'
# results_path = '/home/hbliu/Video-LLaVA/results/whole_test/videollava_gemini.json'

anno_path = '/home/hbliu/TimeCraft/datasets/nextgqa/sub_anno/llama/sorted_test_rewrite.csv'

annos = pandas.read_csv(anno_path)

# Search item in annos by video_id and q_id
def search_item(video_id, q_id):
    item = annos[(annos['video_id'] == video_id) & (annos['qid'] == q_id)]
    # return the item value of "type"
    return item['type'].values[0]

with open(results_path, 'r') as f:
    results = json.load(f)


per_category = {}

correct_org = 0
correct_r = 0
total = 0
for video, preds in results.items():
    for q_id, pred in preds.items():
        qtype = search_item(int(video), int(q_id))
        if qtype not in per_category:
            per_category[qtype] = {'True':0, 'Total':0}
        per_category[qtype]['Total'] += 1
        total += 1
        # pred_org = options_to_idx[pred['pred'][1:2]]
        pred_r = options_to_idx[pred['pred_r'][1:2]]
        # pred_r = pred['pred_r'][1:2]
        # gt_idx_org = pred['gt_idx']
        gt_idx_r = pred['gt_idx_r']
        # if pred_org == (gt_idx_org):
        #     correct_org += 1
        #     per_category[qtype]['Ori'] += 1
        if pred_r == gt_idx_r:
            correct_r += 1
            per_category[qtype]['True'] += 1

# print(per_category)

print('why', (per_category['CW']['True'])/(per_category['CW']['Total']))
print('before/after', (per_category['TN']['True']+ per_category['TP']['True'])/(per_category['TN']['Total'] + per_category['TP']['Total']))
print('how', (per_category['CH']['True'])/(per_category['CH']['Total']))
print('when', (per_category['TC']['True'])/(per_category['TC']['Total']))
breakpoint()

# For llama rewrite
# correct_org = 374 (56.32%), correct_r = 240 (36.14%), total = 664

# For puyu rewrite
# correct_org

# For whole test set
# correct = 3249 (58.50%), total = 5553