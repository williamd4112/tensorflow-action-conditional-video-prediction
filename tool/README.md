# How to prepare training data

1. Modify your agent code
2. Integrate [```episode_collector```](episode_collector.py) with your code. The following is an short example.
```python
from episode_collector import EpisodeCollector
import gym
import cv2
import os
import numpy as np

env = gym.make('Pong-v0')

preprocess_func = lambda x: cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis], [84, 84, 1])

for episode in range(10):
  collector = EpisodeCollector(path=os.path.log('Pong-v0', '%04d.tfrecords' % (episode)), 
                          preprocess_func=preprocess_func, 
                          skip=4)
  done = False
  observation = env.reset()
  state = np.zeros([84, 84, 4], dtype=np.float32)
  state[:, :, -1:] = preprocess_func(observation)
  while not done:
    action = policy(state)
    observation, reward, done, _ = env.step(action)
    collector.save(s=state, a=action, x_next=observation)
    
    state[:,:,:-1] = state[:,:,1:]
    state[:,:,-1:] = preprocess_func(observation)
  collector.close()
```
3. After step 2, you shold have some data like the following:
```
Pong-v0/0000.tfrecords
Pong-v0/0001.tfrecords
Pong-v0/0002.tfrecords
Pong-v0/0003.tfrecords
...
Pong-v0/0100.tfrecords
```
4. Use ```compute_mean.py``` to compute image mean
```
python compute_mean.py Pong-v0 Pong-v0/mean.npy
```

5. Split your training data into two directory, train and test
```
Pong-v0/
  train/0000.tfrecords
  train/0001.tfrecords
  train/0002.tfrecords
  train/0003.tfrecords
  ...
  train/0100.tfrecords
Pong-v0/
  test/0101.tfrecords
  test/0102.tfrecords
  test/0103.tfrecords
  test/0104.tfrecords
  ...
  test/0105.tfrecords
```

NOTE: If your directory structure is same as the above, you can simply use ```train_atari.sh``` to train your agent. (Usage of ```train_atari.sh``` in [here](../README.md))
