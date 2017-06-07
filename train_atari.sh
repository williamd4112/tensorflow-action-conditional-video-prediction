GAME=$1
NUM_ACT=$2
GPU=$3
TRAIN="${GAME}/train"
TEST="${GAME}/test"
MEAN="${GAME}/mean.npy"
LOG="models/${GAME}-model"

export CUDA_VISIBLE_DEVICES=$3
python train.py --train ${TRAIN} --test ${TEST} --mean ${MEAN} --num_act ${NUM_ACT} --log ${LOG} 
