# Installtion
```
cd tensorflow-action-conditional-video-prediction
source setup.sh
```

# Train
## Atari
```
./train_atari.sh ${game name} ${num_act} ${colorspace [rgb|gray]} {gpu id}
e.g. ./train_atari.sh MsPacman-v0 9 gray 0

You can use tensorboard to monitor training process.
e.g. tensorboard --port 6006 --logdir=train:models/Pong-v0/train,test:models/Pong-v0/test

NOTE: I don't test model during training currently.
```

# Test

# Model zoo
Since size of model is too large, please download pretrained models from [here](https://drive.google.com/drive/u/0/folders/0B5wysG7CaEswVnNJdUkyZ29DR2s)
