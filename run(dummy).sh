#!/bin/bash
#################################################################
# A40 1GPU Job Script for HPC System "KAGAYAKI" 
#                                       2022.3.3 k-miya
#################################################################

#PBS -N chokosen-loveti
#PBS -j oe
#PBS -q GPU-S
#PBS -l select=2:ngpus=1

cd [your clone dir]

source /etc/profile.d/modules.csh

module load singularity
mkdir -p ./tmp
chmod 755 ./tmp

echo "==== make singularity container ====="

singularity exec --nv --bind ./tmp:/container/tmp ./singularity/python.sif /bin/bash  <<EOF

echo "==== pip install ====="
pip install --upgrade pip
pip install --target ./.local -r requirements.txt

echo "==== change dir and clear logs ====="

rm -rf ./log
rm -rf ./callback
# export PATH=$PATH:./.local/bin
export PYTHONPATH=$PYTHONPATH:./src
source ./training_config.sh

echo "==== run python ====="
python ./main.py

EOF
