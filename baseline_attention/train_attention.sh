#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# 固定参数
train_epochs=50
batch_size=16
dataset_path='./dataset_attention/'

# 循环变量
D_embeds=(256 512)
S_TFs=(10)
Ks=(10)
H_TFs=(6 12)
learning_rates=(0.0001 0.001)

for D_embed in "${D_embeds[@]}"
do
  for S_TF in "${S_TFs[@]}"
  do
    for K in "${Ks[@]}"
    do
      for H_TF in "${H_TFs[@]}"
      do
        for learning_rate in "${learning_rates[@]}"
        do
          echo "Running model training with the following parameters:"
          echo "D_embed: $D_embed"
          echo "S_TF: $S_TF"
          echo "K: $K"
          echo "H_TF: $H_TF"
          echo "Learning rate: $learning_rate"
          echo "Epochs: $train_epochs"
          echo "Batch size: $batch_size"
          echo "Dataset path: $dataset_path"

          python ./baseline_attention/train_attention.py \
            --d_embed $D_embed \
            --s_tf $S_TF \
            --k $K \
            --h_tf $H_TF \
            --learning_rate $learning_rate \
            --train_epochs $train_epochs \
            --batch_size $batch_size \
            --dataset_path $dataset_path \

        done
      done
    done
  done
done
