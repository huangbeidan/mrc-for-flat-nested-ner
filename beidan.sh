#!/bin/bash

export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

DATA_DIR="mrc/conll03"
BERT_DIR="mrc/bert-large"

SPAN_WEIGHT=0.1
DROPOUT=0.2
LR=8e-6
MAXLEN=128

OUTPUT_DIR="mrc/train_logs/en_coll03/en_coll03_bertlarge_lr${LR}20201119_dropout${DROPOUT}_bsz16_maxlen${MAXLEN}"

## 如果是chinese记得补上 --chinese
python3 trainer.py \
--data_dir $DATA_DIR \
--bert_config_dir $BERT_DIR \
--max_length $MAXLEN \
--batch_size 4 \
--gpus "1" \
--precision=16 \
--progress_bar_refresh_rate 1 \
--lr $LR \
--distributed_backend=ddp \
--val_check_interval 0.5 \
--accumulate_grad_batches 1 \
--default_root_dir $OUTPUT_DIR \
--mrc_dropout $DROPOUT \
--max_epochs 20 \
--weight_span $SPAN_WEIGHT \
--span_loss_candidates "pred_and_gold" \
--loss_type "adaptive_dice"
