#!/bin/csh
csh << EOF
source /etc/profile.d/modules.csh
module load singularity
mkdir singularity
singularity pull ./singularity/cafeobj.sif docker://namelessogya/cafeobj
EOF