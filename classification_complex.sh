#!/bin/bash
#SBATCH --job-name="classification_complex"
#SBATCH --output=job-%j.txt
#SBATCH --error=job-%j.txt
#SBATCH --time=12:00:00
#SBATCH -n 4
#SBATCH --mem=20G

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=email address


module load cuda/10.1.105 python/3.6.6

export PATH="/users/ydai38/.local/bin:$PATH"


python3 -u classification_complex.py -ne 1000 -lr 0.1 -reg 0.03 -dataset Deepddi -emb_dim 200 -neg_ratio 1 -batch_size 1024 -save_each 100 -discriminator_range 1 | tee ./output.txt