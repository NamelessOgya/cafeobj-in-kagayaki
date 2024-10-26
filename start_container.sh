source /etc/profile.d/modules.csh
module load singularity
mkdir -p ./tmp
chmod 755 ./tmp

echo "==== dive into container ====="
singularity shell --bind ./tmp:/container/tmp ./singularity/cafeobj.sif