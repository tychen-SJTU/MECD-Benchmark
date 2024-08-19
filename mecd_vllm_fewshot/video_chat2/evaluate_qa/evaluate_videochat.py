import json

# Load dict from json file


options_to_idx = {'A': 1, 'B':2, 'C':3, 'D':4, 'E':5}

results_path = 'your_path/Ask-Anything/video_chat2/results/videochat_puyu_results_all_test_r.json'

with open(results_path, 'r') as f:
    results = json.load(f)


correct_org = 0
correct_r = 0
total = 0
for video, preds in results.items():
    for q_id, pred in preds.items():
        total += 1
        pred_org = options_to_idx[pred['pred'][1:2]]
        pred_r = options_to_idx[pred['pred_r'][1:2]]
        gt_idx_org = pred['gt_idx']
        gt_idx_r = pred['gt_idx_r']
        if pred_org == (gt_idx_org):
            correct_org += 1
        if pred_r == gt_idx_r:
            correct_r += 1
breakpoint()

# For llama rewrite
# correct_org = 374 (56.32%), correct_r = 240 (36.14%), total = 664

# For puyu rewrite
# correct_org

# For whole test set
# correct = 3249 (58.50%), total = 5553