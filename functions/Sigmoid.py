import numpy as np


class Sigmoid():
  def __call__(self, x: np.ndarray):
    y = 1 / (1 + np.exp(-x))
    return y
  
  def derive(self, y_o: np.ndarray):
    y = self(y_o) * (1 - self(y_o))
    return y
