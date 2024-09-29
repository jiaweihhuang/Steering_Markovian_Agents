import pickle
import numpy as np
import os

class Logger:
    def __init__(self, num_players):
        self.num_players = num_players
        self.data = []

    def record(self, trial):
        self.data.append(trial)

    def save(self, path):
        parent_dir = '/'.join(path.split('/')[:-1])
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        with open(path, 'wb') as f:
            pickle.dump(self, file=f)


    def load(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)