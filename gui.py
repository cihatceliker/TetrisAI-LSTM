import time
import threading
import numpy as np
import random
import pickle
from environment import Environment, ALL_SHAPES
from tkinter import Frame, Canvas, Tk
from agent import Agent, load_agent
import sys
from pyscreenshot import grab
#import pyautogui
import pickle

COLORS = {
    0: "#fff",    # BACKGROUND
    2: "#3e2175", # SHADOW
    1: "#c2abed", # CURRENT PIECE
    3: "#5900ff"  # GROUND
}

class GameGrid():

    def __init__(self, speed=0.02, size=720):
        self.draw_next_offset = size/4
        width = size / 2
        height = size
        self.root = Tk()
        self.root.configure(background=COLORS[0])
        self.game = Canvas(self.root, width=width, height=height, bg=COLORS[0])
        self.game.pack()
        self.env = Environment()
        self.env.reset()
        self.speed = speed
        self.size = size
        self.rectangle_size = size/self.env.row
        self.pause = False
        self.image_counter = 0
        self.commands = {
            113: 1, # Left
            114: 2, # Right
            53: 3, # Z
            52: 4, # X
            65: 5, # Drop
            37: 0 # Do nothing
        }
        self.init()
        self.root.title('Tetris')

        if 3 == len(sys.argv):
            history = load_agent(sys.argv[1])
            self.processed = []
            for state,_,_,_ in history:
                self.processed.append(self.process_channels(state))
            threading.Thread(target=self.watch_history).start()
        else:
            self.agent = Agent(6) if len(sys.argv) == 1 else load_agent(sys.argv[1])
            print("duration",max(self.agent.durations))
            print("score",max(self.agent.scores))
            threading.Thread(target=self.watch_play).start()
        self.root.mainloop()

    def process_channels(self, obs):
        board_repr = np.zeros((20,10))
        board_repr[obs[2]==1] = 2
        board_repr[obs[1]==1] = 1
        board_repr[obs[0]==1] = 3
        return board_repr
            
    def watch_play(self):
        while True:
            duration = 0
            done = False
            state, next_piece = self.env.reset()
            while not done:
                action = self.agent.select_action(state, next_piece)
                state, reward, done, next_piece = self.env.step(action)
                self.board = self.process_channels(state)
                self.update()
                duration += 1
                time.sleep(self.speed)
            print("duration", duration)

    def update(self):
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                rect = self.game_area[i][j]
                curr = int(self.board[i, j])
                color = COLORS[curr]
                self.game.itemconfig(rect, fill=color)

    def watch_history(self):
        for state in self.processed:
            self.board = state
            self.update()
            #self.take_screenshot()
            time.sleep(self.speed)

    def init(self):
        def draw(x1, y1, sz, color, func):
            return func(x1, y1, x1+sz, y1+sz, fill=color, width=0)
        self.game_area = []
        for i in range(self.env.row):
            row = []
            for j in range(self.env.col):
                color = COLORS[0]
                rect = draw(j*self.rectangle_size, i*self.rectangle_size, 
                            self.rectangle_size, color, self.game.create_rectangle)
                row.append(rect)
            self.game_area.append(row)


if __name__ == "__main__":
    GameGrid()
