sbatch  \
         --job-name=train_AE \
         --partition=gpu-short \
         --gres=gpu:1 \
         --mem-per-gpu=11GB \
         --ntasks=2 \
         --nodes=1 \
         --time=2:00:00 \
         /home/s1825216/git/MRP/src/neural_nets/AE/scripts/train_AE.sh
