#!/bin/bash

python train_mlm.py \
    --model_name_or_path Modelos/epocas_HelBERT-base-uncased-scratch/checkpoint-epoca-20 \
    --tokenizer_name Tokenizadores/HelBERT-base-uncased-scratch \
    --train_file Datasets/HelBERT-base-uncased-scratch/treino.txt \
    --validation_file Datasets/HelBERT-base-uncased-scratch/teste.txt \
    --output_dir Modelos/HelBERT-base-uncased-scratch \
    --overwrite_output_dir false \
    --do_train \
    --do_eval \
    --do_predict \
    --line_by_line \
    --fp16 \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --preprocessing_num_workers 16 \
    --max_seq_length 128 \
    --gradient_accumulation_steps 8 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 0.000001 \
    --learning_rate 0.0001 \
    --weight_decay 0.01 \
    --warmup_steps 10000 \