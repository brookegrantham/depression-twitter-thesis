#!/bin/bash
#
#
#SBATCH --job-name=electra-pretrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=50GB
#SBATCH --output=/user/home/bg17893/electra.log

export CORPUS_DIR =
export VOCAB_FILE =
export OUTPUT_DIR =
export DATA_DIR =
export MODEL_NAME =

python build_pretraining_dataset.py \
  --corpus-dir $CORPUS_DIR \
  --vocab-file $VOCAB_FILE \
  --output-dir $OUTPUT_DIR \
  --max-seq-length 128 \
  --num-processes 1 \
  --blanks-seperate-docs True

python run_pretraining.py \
  --data-dir $DATA_DIR \
  --model-name $MODEL_NAME \ '''electrasmall look in notes to pretrain from another model'''
  --hparams {"num_train_steps":4010000}







