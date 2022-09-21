#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --job-name=pretrain_roberta
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=6
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:0
#SBATCH --mem=5000M

for SPLIT in train valid test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done


fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref wikitext-103-raw/wiki.train.bpe \
    --validpref wikitext-103-raw/wiki.valid.bpe \
    --testpref wikitext-103-raw/wiki.test.bpe \
    --destdir data-bin/wikitext-103 \
    --workers 60

fairseq-hydra-train -m --config-dir examples/roberta/config/pretraining \
--config-name base task.data=$DATA_DIR

checkpoint.restore_file=/path/to/roberta.base/model.pt

dataset.batch_size=8




