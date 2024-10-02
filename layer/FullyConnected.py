import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
  @abstractmethod
  def __call__(self, input: np.ndarray):
    pass

  @abstractmethod
  def init_params(self, input: np.ndarray):
    pass
  
  @abstractmethod
  def train(self, output_deltas: np.ndarray, data_index: int):
    pass



class FFLayer(Layer):
  def __init__(self, units: int, bias: bool, function):
    self.units = units
    self.bias = bias
    self.func = function
    self.inited = False
    
    self.weights = None
    self.inputs = None
    self.outputs = None
    self.data_index = 0


  def init_params(self, input: np.ndarray, batch_size: int):
    input_len = len(input)
    if self.bias:
      input_len += 1
      
    self.batch_size = batch_size
    self.weights = np.random.rand(self.units, input_len) - 0.5
    self.inited = True
    out = self(input, init=True)
    out_len = len(out)
    
    self.inputs = np.zeros((self.batch_size, input_len, 1))
    self.outputs = np.zeros((self.batch_size, out_len, 1))
    return out


  def __call__(self, input: np.ndarray, init=False):
    if len(input.shape) != 2 or input.shape[1] != 1:
      raise Exception('invalid input size')
    
    if not self.inited:
      raise Exception('not inited params')

    if self.bias:
      input = np.concatenate((input, [[1.]]), axis=0)
    
    output = np.matmul(self.weights, input)
    activated_output = self.func(output)
    
    if not init:
      self.outputs[self.data_index] = output
      self.inputs[self.data_index] = input
      self.data_index += 1
    return activated_output


  def train(self, output_deltas: np.ndarray, learnig_rate: int):
    if len(output_deltas.shape) != 3 or output_deltas.shape[2] != 1:
      raise Exception('invalid input size')
    self.data_index = 0
    
    derivatives = self.func.derive(self.outputs)
    delta = output_deltas * derivatives
    delta = np.tile(delta, (1, 1, self.weights.shape[1]))
    
    input = np.reshape(self.inputs, (self.batch_size, 1, len(self.inputs[0])))
    input = np.tile(input, (1, self.outputs.shape[1], 1))
    weight_deltas = delta * input * learnig_rate
    weight_deltas = np.sum(weight_deltas, axis=0)
    
    input_deltas = delta * self.weights
    input_deltas = np.sum(input_deltas, axis=1)
    input_deltas = np.reshape(input_deltas, (self.batch_size, len(self.inputs[0]), 1))
    
    if self.bias:
      input_deltas = input_deltas[:, :input_deltas.shape[1] - 1, ...]
    
    self.weights -= weight_deltas
    return input_deltas
  