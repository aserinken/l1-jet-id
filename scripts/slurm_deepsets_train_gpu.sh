#!/bin/sh
#SBATCH --job-name=deepsets_net_train
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=0-01:30
#SBATCH --output=./logs/deepsets_gpu_%j.out

# Set up conda environment and cuda.
source /work/aserinke/miniconda/bin/activate fast_jetclass

while [ $# -gt 0 ]; do
  case "$1" in
    --config=*)
      config="${1#*=}"
      ;;
    --gpu=*)
      gpu="${1#*=}"
      ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument.*\n"
      printf "***************************\n"
      exit 1
  esac
  shift
done

# Run the script with print flushing instantaneous.
export PYTHONUNBUFFERED=TRUE
./deepsets_train --config ${config} --gpu ${gpu}
export PYTHONUNBUFFERED=FALSE
