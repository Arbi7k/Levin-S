#!/bin/bash

#SBATCH --job-name="Midterm Project"
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4GB

# Load the required modules
module load Python
module load CUDA/11.8.0

# Activate your Python environment if needed
source ~/dir01/ENV/bin/activate

# Run your Python script
srun python3 job.py
