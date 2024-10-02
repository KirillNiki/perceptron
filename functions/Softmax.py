import numpy as np
import warnings
warnings.filterwarnings("error")


class Softmax():
  def __call__(self, x: np.ndarray):
    maxx = np.max(x)
    x = x / maxx
    
    y = np.exp(x)
    summ = np.sum(y)
    
    y = y / summ
    return y
  
  def derive(self):
    return 1
