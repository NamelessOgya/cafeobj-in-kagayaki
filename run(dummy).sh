#!/bin/csh
#################################################################
# A40 1GPU Job Script for HPC System "KAGAYAKI" 
#                                       2022.3.3 k-miya
#################################################################

#PBS -N pvp-cartpole
#PBS -j oe
#PBS -q GPU-S
#PBS -l select=2:ngpus=1

source /etc/profile.d/modules.csh
cd [your clone directry]

echo "==== make singularity container ====="
module load singularity
singularity exec --nv --bind ./tmp:/container/tmp ./custom-pytorch-jupyter.sif /bin/bash <<EOF

echo "==== pip install ====="

pip install gymnasium
pip install pettingzoo
pip install --upgrade "ray[rllib]" 
pip install wandb

echo "==== change dir and clear logs ====="
rm -rf ./log
rm -rf ./callback

echo "==== run python ====="
python ./main.py

EOF