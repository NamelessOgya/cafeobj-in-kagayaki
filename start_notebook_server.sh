csh << 'FIRST_EOF'

source /etc/profile.d/modules.csh
module load singularity
mkdir -p ./tmp
chmod 755 ./tmp

echo "==== dive into container ====="
singularity shell --nv --bind ./tmp:/container/tmp ./singularity/python.sif << 'SECOND_EOF'



echo "==== install packages ====="
pip install --upgrade pip
pip install --target ./.local -r requirements.txt

echo "==== register variables ===="
export PATH=$PATH:./.local/bin
export PYTHONPATH=$PYTHONPATH:./src
source ./training_config.sh

jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

