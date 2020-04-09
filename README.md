# TetrisAI-LSTM

Tetris AI using DoubleDQN with LSTM-CNN architecture. In the same amount of training with the [previous version](https://github.com/cihatceliker/TetrisAI), it achieved much better results. The only difference in the model is that the first linear layer is converted to lstm layer. Now because of this, instead of learning from random samples in the memory, it learns from episodes. This led to much smoother gameplay as it can be seen from the gameplays of both.

# 
Gameplay after around 12 hours of training. I'm pretty sure it can be better with some fine-tuning and more training.

![alt text](/tet.gif)
# 