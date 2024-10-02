import matplotlib.pyplot as plt
import numpy as np
import csv

with open('./data/mnist_test.csv', 'r') as file:
  reader = csv.reader(file)

  row = next(reader)[1:]
  row = np.reshape(row, (28, 28))
  row = row.astype(int)
  row = row / 255

plt.imshow(row)
plt.savefig('./test.png')