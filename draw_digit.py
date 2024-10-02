import tkinter as tk
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


line_id = None
line_points = []
line_options = {}
SMALL_SIZE = 28
DIST = 10
SIZE = DIST * SMALL_SIZE
index = 0

def draw_line(event):
    global line_id
    pixelx_st = event.x - (event.x % DIST)
    pixely_st = event.y - (event.y % DIST)
    pixelx_end = pixelx_st + DIST
    pixely_end = pixely_st + DIST
    
    if line_id is not None:
        canvas.delete(line_id)
    line_id = canvas.create_rectangle(pixelx_st, pixely_st, pixelx_end, pixely_end, fill='black', width=0)
    line_id = None


def end_line(event=None):
  global index
  canvas.postscript(file="picture.eps", height=SIZE, width=SIZE, pageheight=SIZE, pagewidth=SIZE, colormode="gray")
  img = Image.open("picture.eps")
  img.save(f'images/image{index}.png', "png")
  canvas.delete('all')
  
  image = Image.open(f'images/image{index}.png')
  pixels = np.array(image.getdata())
  pixels = pixels.reshape((image.size[1], image.size[0], 3))
  pixels = np.sum(pixels, axis=2)
  
  result = np.zeros((SMALL_SIZE, SMALL_SIZE))
  for x in range(SMALL_SIZE):
    for y in range(SMALL_SIZE):
      result[x][y] = 1. if pixels[x*DIST][y*DIST] == 0. else 0.
      
  plt.imsave(f'images_res/image{index}.png', result)
  result = result.reshape(result.shape[0] * result.shape[1], 1)
  np.save(f'images_res/image{index}.npy', result)
  index += 1


root = tk.Tk()
root.geometry(f'{SIZE}x{SIZE}')
root.resizable(False, False)

canvas = tk.Canvas()
canvas.configure(scrollregion=(0, 0, SIZE, SIZE), height=SIZE, width=SIZE)
canvas.pack()

canvas.bind('<Button-1>', draw_line)
canvas.bind('<B1-Motion>', draw_line)
canvas.bind('<ButtonRelease-1>', end_line)

root.mainloop()
