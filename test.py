from main import load_model
import numpy as np
import os

dir_path = '/home/kirill/develop/python/perceptron/images_res'

if __name__ == '__main__':
  file_names = os.listdir(dir_path)
  model = load_model()

  for file_name in file_names:
    if file_name.endswith('.npy'):
      file_path = os.path.join(dir_path, file_name)
      input = np.load(file_path)
      
      output = model(input)
      out = np.where(output == max(output))[0][0]
      print(out)
      