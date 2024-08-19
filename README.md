## 1. Dataset MECD
Our MECD dataset includes 808 and 299 videos for training set and testing set, respectively.
The annotations of training set: `captions/train.json`
The annotations of testing set:  `captions/test.json` 
The videos can be found in ActivityNet official website https://activity-net.org/ according to our provided video ID.

## 2. Training for VGCM
For training our VGCM, run 
```python 
sh scripts/train.sh 
```
#### Benchmark Results 
![Image 1](main.png)

#### Hyperparameters settings
To reproduce our results in the above table, please follow the default hyperparameters settings in: `src/runner.py get_args()`

## 3. Finetuning Video-LLaVA on MECD
We fine-tune Video-LLaVA using LoRA under its official implementation on our entire MECD training set. Please follow the command:
```bash
sh mecd_vllm_finetune/Video-LLaVA-ft/scripts/v1_5/finetune_lora.sh 
```
to finetune Video-LLaVA on our MECD benchmark.

## 4. Few-shot Evaluation of LLM&VLLM
All LLM-based and VLLM-based models are evaluated under a few-shot setting. Specifically, following the approach in causal discovery for NLP tasks, three representative examples are provided during inference. 
#### Video-LLaVA
Please follow the command:
```python
python mecd_vllm_fewshot/Video-LLaVA/videollava/eval/video/run_inference_causal_inference_complete.py
```
to evaulate the causal discovery ability of Video-LLaVA.
#### Videochat2
Similarly, please follow the command:
```python
python mecd_vllm_fewshot/video_chat2/multi_event.py
```
to evaulate the causal discovery ability of Videochat2.
#### GPT-4
Similarly, please follow the command:
```python
python mecd_llm_fewshot/gpt4.py
```
to evaulate the causal discovery ability of GPT-4.
#### Gemini-pro
Similarly, please follow the command:
```python
python mecd_llm_fewshot/gemini.py
```
to evaulate the causal discovery ability of Gemini-pro.

