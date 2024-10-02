from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
  @abstractmethod
  def __call__(self, output: np.ndarray, target: np.ndarray):
    if output.shape != target.shape:
      raise Exception('diffrent shapes')
    
  def derivative(self, output: np.ndarray, target: np.ndarray):
    if output.shape != target.shape:
      raise Exception('diffrent shapes')
  
  
class SquearLoss(Loss):
  def __call__(self, output: np.ndarray, target: np.ndarray):
    super().__call__(output, target)
    
    error = (output - target) ** 2
    return error
  
  def derivative(self, output: np.ndarray, target: np.ndarray):
    super().derivative(output, target)
    
    deriv = (output - target)
    return deriv
