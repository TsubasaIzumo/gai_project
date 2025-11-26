import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

reconstructed = np.load("batch=0_val_generated_images.npy")
print(reconstructed.shape)
# print(reconstructed)

reconstructed_list = np.load("batch=0_val_generated_images.npy").tolist()
print(len(reconstructed_list))
image = reconstructed_list[0]
# print(image)
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='coolwarm',
           vmin=-0.0002, vmax=0.0004)
plt.imshow(image)
plt.colorbar()
plt.show()