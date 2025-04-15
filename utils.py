import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def to_numpy(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1,2,0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image

def show_img(img_2d, title = ''):
    plt.figure(figsize=(6, 6))
    plt.imshow(img_2d)
    plt.title(title)
    plt.show()
