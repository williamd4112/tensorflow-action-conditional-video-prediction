GAME=$1
NUM_ACT=$2
COLOR=$3
GPU=$4
TRAIN="${GAME}/train"
TEST="${GAME}/test"
MEAN="${GAME}/mean.npy"
LOG="models/${GAME}-${COLOR}-model"

export CUDA_VISIBLE_DEVICES=$GPU
python train.py --train ${TRAIN} --test ${TEST} --mean ${MEAN} --num_act ${NUM_ACT} --color ${COLOR} --log ${LOG} 
