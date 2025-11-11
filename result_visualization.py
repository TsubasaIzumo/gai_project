import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

project_folder = "D:\datasets\\radio_images"


# folder = "D1"
# sky_keys = np.load(f"{project_folder}/{folder}/sky_keys.npy")
# phase_centers = np.load(f"{project_folder}/{folder}/ra_dec.npy", allow_pickle=True)
# print(sky_keys)
# print(phase_centers)
# print()
# clean = np.load(f"{project_folder}/{folder}/true/00001.npy")
# plt.title("clean image")
# print(clean)
# plt.figure(figsize=(8, 8))
# plt.imshow(clean, cmap='coolwarm',
#            vmin=-0.0002, vmax=0.0004)
# plt.colorbar()
# plt.show()

reconstructed = np.load("results/experiment1_power2/batch=0_val_generated_images.npy")
print(reconstructed.shape)
print(reconstructed)

reconstructed_list = np.load("results/experiment1_power2/batch=0_val_generated_images.npy").tolist()

image = reconstructed_list[1]
print(image)
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='coolwarm',
           vmin=-0.0002, vmax=0.0004)
plt.imshow(image)
plt.colorbar()
plt.show()


# plt.title("reconstructed image")
# print(clean)
# plt.imshow(clean)
#
# dirty = np.load(f"{project_folder}/{folder}/dirty/00001.npy")
# print("dirty image")
# print(dirty)
# plt.figure(figsize=(8, 8))
# plt.imshow(dirty, cmap='coolwarm',
#            vmin=-0.0002, vmax=0.0004)
# plt.imshow(dirty)
# plt.colorbar()
# plt.show()