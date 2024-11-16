#!/bin/sh
#SBATCH -p dgx2q
#SBATCH --gres=gpu:1




# source D1/mypy311env/bin/activate
# source ./miniconda/bin/activate

# srun jupyter-lab

# srun python3 code/Bi-LSTM.py
# srun python3 code/Classical.py
# srun python3 code/Transformers.py
 srun python3 code/service.py
