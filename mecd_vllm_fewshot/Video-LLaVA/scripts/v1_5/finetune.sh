

JSON_FOLDER="datasets/train_json"
IMAGE_FOLDER="llava_all_image_video"
VIDEO_FOLDER="datasets"
cd your_path/Video-LLaVA/

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed videollava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path videollava/train/cache_dir/e16778e47e589512d7e69f964850c8cad775a335 \
    --version llava_sg_video \
    --data_path datasets/ActivityNet_QA/annotations/train_sg_grounding_DINO_filtered.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_folder datasets/ActivityNet_QA/train \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/videollava-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir ./cache_dir \
    --scene_graph True
