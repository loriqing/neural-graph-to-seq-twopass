#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -C K80 --time=1:00:00 --output=decode.out --error=decode.err
#SBATCH --mem=10GB
#SBATCH -c 6

export PYTHONPATH=$PYTHONPATH:/home/liuqing/AMR2TEXT/neural-graph-to-seq-mp

python src_g2s/G2S_beam_decoder.py --model_prefix logs_g2s_gold/G2S.gold.allbpe5000lr10e3 \
        --in_path data/all-dev-bpe5000.json \
        --out_path logs_g2s_gold/dev.g2s.gold.allbpe5000lr10e3\.tok \
        --mode beam \
        --device_id $1
