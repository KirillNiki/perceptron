from losses.Loss import Loss
import numpy as np
from tqdm import tqdm
import sys


class Model():
  def __init__(self, layers: list):
    self.layers = layers
    self.losses = list()
    self.inited = False
    self.metrix_history = \
    {
      'accuracy': [], 
      'loss': [],
    }
    
    
  def init_weights(self, input: np.ndarray, batch_size: int):
    self.batch_size = batch_size
    for layer in self.layers:
      input = layer.init_params(input, batch_size)
    
    self.inited = True
    
    
  def __call__(self, input: np.ndarray, validation=False):
    if not self.inited:
      raise Exception('weights not initialized')
    
    for layer in self.layers:
      input = layer(input, init=validation)
    return input
  
  
  def accur(self):
    if len(self.losses) == 0:
      raise Exception('len of losses 0')
    
    correct_count = 0
    for loss in self.losses:
      if loss <= 0.05:
        correct_count += 1
      
    accur = correct_count / len(self.losses)
    self.metrix_history['accuracy'].append(accur)
    return accur
  
  
  def loss(self):
    if len(self.losses) == 0:
      raise Exception('len of losses 0')
    
    summ = sum(self.losses)
    loss = summ / len(self.losses)
    
    self.metrix_history['loss'].append(loss)
    return loss
  
  
  def check_losses(self):
    stop_train_counter = 15
    stop_train_loss = 0.01
    
    if len(self.metrix_history['loss']) < stop_train_counter:
      return False
    
    counter = 0
    stop_train = False
    for i in range(len(self.metrix_history['loss'])-1, -1, -1):
      loss = self.metrix_history['loss'][i]
      
      if loss <= stop_train_loss:
        counter += 1
      
      if counter == stop_train_counter:
        stop_train = True
        break
    return stop_train

  
  def fit(self, 
          loss_func: Loss, 
          train_ds,
          val_ds,
          out_shape,
          learnig_rate: float,
          epochs: int,
          dataset_len: int,
          val_steps: int,
          metrics: list = [],
          ):
    for epoch in tqdm(range(epochs)):
      losses = np.zeros((self.batch_size, out_shape, 1))
      loss_index = 0
      
      for i in range(1, dataset_len + 1):
        input, target = next(train_ds)
        output = self(input)
        losses[loss_index] = loss_func.derivative(output, target)
        loss_index += 1
        
        if i % self.batch_size == 0 or i == dataset_len:
          for j in range(len(self.layers), 0, -1):
            index = j - 1
            losses = self.layers[index].train(losses, learnig_rate)
          losses = np.zeros((self.batch_size, out_shape, 1))
          loss_index = 0
      
      
      self.losses = list()
      for val_step in range(val_steps):
        input, target = next(val_ds)
        output = self(input, validation=True)
        
        calculated_losses = loss_func(output, target)
        loss = sum(calculated_losses)
        self.losses.append(loss[0])

      
      print(f'epoch: {epoch} >>>>>')
      sys.stdout.flush()
      
      for metric in metrics:
        if metric == 'accuracy':
          metric_output = self.accur()
        elif metric == 'loss':
          metric_output = self.loss()
          
          if metric_output is None:
            raise Exception('none loss')
          
        print(f'{metric}: {metric_output}')
        sys.stdout.flush()
        
      stop_train = self.check_losses()
      if stop_train:
        break
      