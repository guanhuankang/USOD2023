#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=4        # 4 OpenMP threads on EACH MPI TASK
#SBATCH --gres=gpu:1             # Using 1 GPU card
#SBATCH --mem=64GB               # Request 50GB memory
#SBATCH --time=0-11:59:00        # Time limit day-hrs:min:sec
#SBATCH --output=output.log   # Standard output
#SBATCH --error=error.err    # Standard error log

export envname=burgundy
echo "["+$envname+"] working dir:"
pwd
nvidia-smi
python searcher.py