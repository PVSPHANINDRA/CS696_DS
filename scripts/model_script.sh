#!/bin/bash
#SBATCH -c 16  # Number of Cores per Task
#SBATCH --mem=250000  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH --gres=gpu:1  # Number of GPUs
#SBATCH -t 04:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

# Replace this with the work path of your user till the github repository
cd /work/pi_adrozdov_umass_edu/syerawar_umass_edu/696DS
module load conda

#Initialize Environment
conda activate ../envs/vadops

#cd ./TextClassification

bash ./scripts/run.sh
