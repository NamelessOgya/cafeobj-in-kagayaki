#!/bin/csh
csh << EOF
source /etc/profile.d/modules.csh
module load singularity
singularity pull ./singularity/python.sif docker://pytorch/pytorch 
EOF