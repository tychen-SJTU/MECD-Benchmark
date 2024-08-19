CUDA_VISIBLE_DEVICES=4 python3 src/runner.py \
--w_CEloss 3.0 \
--w_hybrid_loss1 5.0 \
--mix_weight 0.0 \
--max_v_len 50 \
--loss_aux_weight2 2e-6 \
--w_compensate 0.0 \
--hybrid_quant 0.065 \
--logits_mix_weight 0.0015 \
--del_words 0 \
--mask_frames 0 \
--multi_chains_k 3 \
--multi_chains_b 1 \

