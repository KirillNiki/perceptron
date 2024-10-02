from main import load_model, test_data_gen
import numpy as np


if __name__ == '__main__':
  with open('/home/kirill/develop/python/perceptron/data/mnist_test.csv', 'r') as file:
    reader = file.readlines()
  test_len = len(reader)
  
  val_ds = test_data_gen()
  model = load_model()
  count = 0
    
  for i in range(test_len):
    input, target = next(val_ds)
    output = model(input, validation=True)
    
    tar = np.where(target == max(target))[0][0]
    out = np.where(output == max(output))[0][0]
    if tar == out:
      count += 1
  
  print(count / test_len)
  