# TetrisAI-LSTM


## Running the code
PyTorch is needed to run the code. Watch the trained model using:
```
python gui.py trained_model.tt
```
Or train an agent by running the train.py file.
# 

Tetris AI using DoubleDQN with LSTM-CNN architecture. In the same amount of training with the [previous version](https://github.com/cihatceliker/TetrisAI), it achieved much better results. The only difference in the model is that the first linear layer is converted to lstm layer. Now because of this, instead of learning from random samples in the memory, it learns from episodes. This led to much smoother gameplay as it can be seen from the gameplays of both.

# 
Input has a shape of (4,20,10). Each tile is encoded along the depth of 4 depending on whether its a part of the current piece, ground piece, shadow or if it's empty or not. Shadow is a projection of the current piece to the ground.

# 
Here is a summary of the model
```


                                      ----- Input --
                                    /         |      \
                                   V          |       \
            Conv2d(4, 16, 5, padding=2)       |        \
                        |                     |         \
                        V                     |          \
            Conv2d(16, 24, 3, padding=1)      |           \  
                        |                     V            V
                        V
    NextPieceInfo - MaxPool2d(2) - Conv2d(4, 8, (20,1)) - Conv2d(4, 8, (1,10)) 
        
                    Concatenation
                        |
                        V
                LSTM(1447, 256)
                Linear(256, 6) -> Actions

```
# 
An example gameplay(cherry-picked) after around 12 hours of training.

![alt text](/tet.gif)
# 

