from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained(
    'lmsys/vicuna-7b-v1.5',
    padding_side="right",
    use_fast=False,
)

# tokenizer.add_special_tokens({'additional_special_tokens': ['<entity>']})
tokenizer.add_tokens('[entity]')


breakpoint()
