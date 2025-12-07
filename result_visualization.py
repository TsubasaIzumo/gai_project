import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

reconstructed = np.load("E:\\astroDDPM\generated_pretrained_20251205_190211_power2\\batch=0044_sample=0222_run=03_latent=04.npy")
print(reconstructed.shape)
# print(reconstructed)

reconstructed_list = np.load("E:\\astroDDPM\generated_pretrained_20251205_190211_power2\\batch=0044_sample=0222_run=03_latent=04.npy").tolist()
print(len(reconstructed_list))
image = reconstructed_list[0]
# print(image)
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='coolwarm',
           vmin=-0.0002, vmax=0.0004)
plt.imshow(image)
plt.colorbar()
plt.show()