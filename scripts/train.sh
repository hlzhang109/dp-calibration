#!/bin/bash

learningrate=5e-4 
dataset=mnli
bs=20 
gradient_accumulation_steps=300 # 300 qqp # 300 mnli # 100 qnli # 50 sst-2
norm=0.1
eps=8
epoch=18 #18 # qqp - 18 # sst-2 - 3 # mnli - 18 # qnli - 6 
lr_decay=yes
seed=42
wd=0
eval_steps=20
num_labels=3
valid_ratio=0.1

res_path=ablation
al_strategy=EntropySampling
non_private=no
noclip=True
model_name=${dataset^^}_noclip${norm}_dp_epoch${epoch}_lr${learningrate} 
outpath=${res_path}/${model_name} 
corr_prob=0
dropout_prob=0

export PYTHONPATH="."
cmd="python classification/run_classification.py \
    --non_private ${non_private} \
    --noclip ${noclip} \
    --al False \
    --full_training True \
    --train_size 300000 --n_init_labeled 100000 \
    --n_round 10 --n_query 10000 \
    --strategy ${al_strategy} \
    --task_name ${dataset} --data_dir classification/data/original/${dataset^^} \
    --save_logit True \
    --do_train True \
    --do_eval True \
    --save_logit_dir $outpath \
    --overwrite_output_dir \
    --model_name_or_path roberta-base \
    --few_shot_type finetune \
    --num_k 1 \
    --num_sample 1 --seed ${seed} \
    --template *cls**sent-_0*?*mask*,*+sentl_1**sep+* \
    --num_train_epochs ${epoch} --target_epsilon ${eps} \
    --per_device_train_batch_size ${bs} \
    --per_device_eval_batch_size 8 \
    --per_example_max_grad_norm ${norm} --ghost_clipping False \
    --adam_epsilon 1e-08 \
    --weight_decay ${wd} --max_seq_len 256 --evaluation_strateg steps --eval_steps ${eval_steps} \
    --evaluate_before_training True --first_sent_limit 200 \
    --other_sent_limit 200 --truncate_head yes \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learningrate} --lr_decay ${lr_decay} \
    --evaluate_after_training True \
    --corruption_prob ${corr_prob} --corruption_type unif \
    --num_labels ${num_labels} \
    --dropout_prob ${dropout_prob} \
    --output_dir $outpath" 

myrun="" # Customized command in your cluster to submit jobs

echo %{conda info --envs}

CMD="${myrun} \" ${cmd} "\"
eval ${CMD}