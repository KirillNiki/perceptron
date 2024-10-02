from layer.Model import Model
from layer.FullyConnected import FFLayer
from functions.ReLU import ReLU
from functions.Sigmoid import Sigmoid
from losses.Loss import SquearLoss

import sys
import csv
import numpy as np
import pickle
from matplotlib import pyplot as plt
import os
from dotenv import load_dotenv 
load_dotenv()


PATHS = {
  'train': 'data/mnist_train.csv',
  'test': 'data/mnist_test.csv',
}
MODEL_PATH = os.getenv('MODEL_PATH')
METRIX_FOLDER = os.getenv('METRIX_FOLDER')

EPOCHS = int(os.getenv('EPOCHS'))
INPUT_SHAPE = int(os.getenv('INPUT_SHAPE'))
OUTPUT_SHAPE = int(os.getenv('OUTPUT_SHAPE'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
VAL_STEPS = int(os.getenv('VAL_STEPS'))


def train_data_gen():
  while True:
    csv_path = PATHS['train']
    with open(csv_path, newline='') as csvfile:
      reader = csv.reader(csvfile)

      for row in reader:
        target = np.zeros((OUTPUT_SHAPE, 1))
        tar_index = int(row[0])
        target[tar_index][0] = 1.
        
        input = np.array(row[1:])
        input = input.astype(np.float64) / 255
        input = input.reshape(len(input), 1)
        yield input, target


def test_data_gen():
  while True:
    csv_path = PATHS['test']
    with open(csv_path, newline='') as csvfile:
      reader = csv.reader(csvfile)

      for row in reader:
        target = np.zeros((OUTPUT_SHAPE, 1))
        tar_index = int(row[0])
        target[tar_index][0] = 1.
        
        input = np.array(row[1:])
        input = input.astype(np.float64) / 255
        input = input.reshape(len(input), 1)
        yield input, target


def init_model():
  model = Model([
    FFLayer(512, bias=True, function=Sigmoid()),
    FFLayer(512, bias=True, function=Sigmoid()),
    FFLayer(OUTPUT_SHAPE, bias=True, function=Sigmoid())
  ])

  input = np.zeros(INPUT_SHAPE)
  input = input.reshape((len(input), 1))
  np.ndarray.fill(input, 0.25)
  
  model.init_weights(input, BATCH_SIZE)
  return model


def save_model(model):
  with open(MODEL_PATH, 'wb') as byte_file:
    pickle.dump(model, byte_file)
    

def load_model():
  with open(MODEL_PATH, 'rb') as byte_file:
    model = pickle.load(byte_file)
  
  return model


def plot_metrics(model: Model):
  for metric in ['accuracy', 'loss']:
    plt.plot(model.metrix_history[metric])
    plt.xlabel(metric)
    plt.savefig(f'{METRIX_FOLDER}/{metric}.png')
    plt.cla()


if __name__ == '__main__':
  with open(PATHS['train'], 'r') as file:
    reader = file.readlines()
  dataset_len = len(reader)
  
  train_ds = train_data_gen()
  test_ds = test_data_gen()
  
  model = init_model()
  model.fit(
    train_ds=train_ds,
    val_ds=test_ds,
    dataset_len=dataset_len,
    loss_func=SquearLoss(),
    out_shape=OUTPUT_SHAPE,
    learnig_rate=LEARNING_RATE,
    epochs=EPOCHS,
    val_steps=VAL_STEPS,
    metrics=['accuracy', 'loss'],
  )
  
  save_model(model)
  plot_metrics(model)
  print('done')
  sys.stdout.flush()
  