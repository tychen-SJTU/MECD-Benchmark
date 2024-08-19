import json

# Load dict from json file

options_to_idx = {'A': '1', 'B':'2', 'C':3, 'D':'4', 'E':'5'}
idx_to_options = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E'}

results_path = 'your_path/Ask-Anything/video_chat2/results/whole_test/videochat2_puyu.json'

with open(results_path, 'r') as f:
    results = json.load(f)


correct_r_s = 0
correct_r = 0
total = 0
for video, preds in results.items():
    for q_id, pred in preds.items():
        total += 1
        pred_r = options_to_idx[pred['pred_r'][1:2]]
        pred_r_s = options_to_idx[pred['pred_r_s'][1:2]]
        # gt_idx_org = options_to_idx[pred['gt_idx']]
        gt_idx_r = pred['gt_idx_r']
        gt_idx_r_s = pred['gt_idx_r_s']
        # gt_idx_r = pred['gt_r_idx']
        if pred_r == (str(gt_idx_r)):
            correct_r += 1
        if pred_r_s == str(gt_idx_r_s):
            correct_r_s += 1
print(correct_r)
print(correct_r_s)
breakpoint()


# >>>>> For video chat 
# For llama rewrite
# correct_org = 374 (56.32%), correct_r = 240 (36.14%), total = 664

# For puyu rewrite
# correct_org

# For whole test set
# correct = 3249 (58.50%), total = 5553

# For the gemini rewrite whole set
# correct = 2041 (36.76%), correct_r = 1997, total = 5549

# For the llama rewrite whole set
# correct = 2154 (38.82%), correct_r = 2114, total = 5549

# For the puyu rewrite whole set
# correct = 1705 (30.74%), correct_r = 1695, total = 5549


# >>> For videollava
# For llama rewrite subset
# correct_org = 381 (57.38%), corect_r = 286 (43.07%), total = 664

# For puyu rewrite subset
# correct_org = 382 (57.53%), corect_r = 246 (37.05%), total = 664

# For whole test set
# correct = 3235 (58.25%), total = 5553

# For the gemini rewrite whole set
# correct = 2444 (44.04%), correct_r = 2458, total = 5549

# For the llama rewrite whole set
# correct = 2896 (52.18%), correct_r = 2846, total = 5549

# For the puyu rewrite whole set
# correct = 1897 (34.21%), correct_r = 1897, total = 5549

