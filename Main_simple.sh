#!/bin/bash
#SBATCH --job-name="aaeSimple"
#SBATCH --output=job-%j.txt
#SBATCH --error=job-%j.txt
#SBATCH --time=48:00:00
#SBATCH -n 8
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=60G


#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=email adress


module load cuda/10.1.105 python/3.6.6

export PATH="/users/ydai38/.local/bin:$PATH"

python3 -u main_simple.py -ne 1000 -D_lr 0.1 -G_lr 0.001 -reg 0.3 -dataset Deepddi -emb_dim 200 -neg_ratio 1 -batch_size 512 -save_each 100 -discriminator_range 1 | tee ./output.txt