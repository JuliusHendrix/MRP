sbatch \
  --job-name=train_MRAE_gpu \
  --partition=gpu-long \
  --gres=gpu:1 \
  --mem-per-gpu=11G \
  --ntasks=1 \
  --nodes=1 \
  --time=168:00:00 \
  --mail-user=hendrix@strw.leidenuniv.nl \
  --mail-type=ALL \
  --output=train_MRAE_gpu.out \
  $HOME/git/MRP/src/neural_nets/individualAEs/MRAE/train_MRAE.sh