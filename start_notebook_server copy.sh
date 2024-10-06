csh << 'FIRST_EOF'

source /etc/profile.d/modules.csh
module load singularity
mkdir -p ./tmp
chmod 755 ./tmp

echo "==== dive into container ====="
singularity shell --nv --bind ./tmp:/container/tmp ./singularity/python.sif << 'SECOND_EOF'
pwd

echo "==== install packages ====="
pip install --upgrade pip
pip install gymnasium
pip install pettingzoo
pip install --upgrade "ray[rllib]" 
pip install wandb
pip install jupyter

echo "==== register variables ===="
export PATH=$PATH:/home/s2430014/.local/bin

jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

