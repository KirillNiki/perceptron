import numpy as np
import numpy.ma as ma


class ReLU():
  def __call__(self, x: np.ndarray):
    y = np.copy(x)
    y[y < 0] = 0
    return y
  
  def derive(self, x):
    y = np.copy(x)
    y[y >= 0] = 1
    y[y < 0] = 0
    return y
