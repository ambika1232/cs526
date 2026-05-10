#!/bin/bash
#SBATCH --job-name=coalescing_gpu
#SBATCH --output=coalescing_gpu_%j.out
#SBATCH --error=coalescing_gpu_%j.err

#SBATCH --partition=IllinoisComputes (or IllinoisComputes-GPU)

#SBATCH --account=ambikas2-ic


#SBATCH --time=00:30:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH --gres=gpu:1
# If your cluster uses a specific GPU type, use something like:
# #SBATCH --gres=gpu:a100:1
# or
# #SBATCH --gpus=1

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA devices:"
nvidia-smi

# Load modules as needed on your cluster
# module purge
# module load cuda
# module load llvm
# module load cmake

# Go to your project
cd /path/to/your/project/build

# Rebuild if needed
cmake --build . -j

# Run your analysis pass
opt -load-pass-plugin ./CoalescingPass.dylib \
    -passes="coalescing-pass" \
    -disable-output 2mm.ll

# If you also have a GPU executable/kernel benchmark, run it here:
# ./your_gpu_program