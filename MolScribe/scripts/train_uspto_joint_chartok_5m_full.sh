#!/bin/bash
# Changed for core
NUM_NODES=4
NUM_GPUS_PER_NODE=8
NODE_RANK=${RANK}

BATCH_SIZE=352
ACCUM_STEP=1

MASTER_ADD=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
# $(shuf -n 1 -i 10000-65535)
DATESTR=$(date +"%m-%d-%H-%M")
BASE_MOUNT=${molscribe_data} # Get the base path from blob mount
SAVE_PATH="${BASE_MOUNT}/output/default_training/swin_base_char_aux_1m680k"
# mkdir -p ${SAVE_PATH}
# az://scratch@stmlwsimlproduse001.blob.core.windows.net/projects/cdd/
set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr=$MASTER_ADD --master_port $MASTER_PORT \
    ml_projects/chem_data_extraction/MolScribe/train.py \
    --data_path "${BASE_MOUNT}/data" \
    --train_file full_compiled/all_canonicalized.txt \
    --aux_file uspto_mol/train_680k.csv --coords_file aux_file \
    --valid_file real/USPTO.csv \
    --test_file real/CLEF.csv,real/UOB.csv,real/USPTO.csv,real/staker.csv,real/acs.csv \
    --vocab_file ml_projects/chem_data_extraction/MolScribe/molscribe/vocab/vocab_chars.json \
    --formats chartok_coords,edges \
    --dynamic_indigo --augment --mol_augment \
    --include_condensed \
    --coord_bins 64 --sep_xy \
    --input_size 384 \
    --encoder swin_base \
    --decoder transformer \
    --encoder_lr 4e-5 \
    --decoder_lr 1e-5 \
    --save_path ${SAVE_PATH} --save_mode all \
    --label_smoothing 0.1 \
    --epochs 100 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps 1 \
    --use_checkpoint \
    --warmup 0.02 \
    --print_freq 20 \
    --num_workers 16 \
    --do_train --do_valid --do_test \
    --custom_enc \
    --train_v1 0 \
    --load_path "${BASE_MOUNT}/output/exp7_learnable_pix_mask_5m_perturb_actual_multi/swin_base_char_aux_1m680k/swin_base_transformer_best.pth" \
    --mask_pos_emb 1 \
    --perturb \
    --do_mops \
    # --data_path_decoder "${BASE_MOUNT}/weights/V1/swin_base_char_aux_1m680k.pth" \
    # --backend nccl 2>&1 \
    # --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    # full_compiled/all_canonicalized.txt