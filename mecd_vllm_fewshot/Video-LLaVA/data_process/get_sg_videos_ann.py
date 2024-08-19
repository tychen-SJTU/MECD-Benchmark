import json

QA_EVAL_PATH = 'eval/GPT_Multi_Videos_QA/Activitynet_Multi_Videos_QA/QA_val_reason_fewshot.json'
GROUNDING_PATH = 'datasets/ActivityNet_QA/annotations/val_sg_grounding_DINO_filtered.json'
OUTPUT_QA_EVAL_PATH = 'eval/GPT_SG_Videos_QA/QA_val_eval.json'


if __name__ == "__main__":
    qa_eval_set = json.load(open(QA_EVAL_PATH, "r"))
    grounding_set = json.load(open(GROUNDING_PATH, "r"))
    
    qa_eval_dict = {}
    for task in qa_eval_set:
        question_id = task['question_id']
        qa_eval_dict[question_id] = task['fewshot_set']
    
    
    new_grounding_dict = {}
    for res in grounding_set:
        question_id = res['question_id']
        new_grounding_dict[question_id] = res
    
    new_grounding_set = []
    for task in grounding_set:
        question_id = task['question_id']
        domain_set = qa_eval_dict[question_id]
        task['fewshot_set'] = domain_set
        new_grounding_set.append(task)
        
    with open(OUTPUT_QA_EVAL_PATH, 'w') as f:
        json.dump(new_grounding_set, f)