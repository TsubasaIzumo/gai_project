import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

project_folder = "D:/datasets/radio_images"

# Function to create a Gaussian
def add_gaussian(array, mean_x, mean_y, std_dev, max_value=1.0):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] += max_value*np.exp(-((i-mean_x)**2 + (j-mean_y)**2)/(2*std_dev**2))
    return array

# Create a blank 512x512 array


# Function to generate random parameters for Gaussian
def generate_random_params():
    mean_x = np.random.randint(64, 512-64)
    mean_y = np.random.randint(64, 512-64)
    std_dev = random.uniform(0.5, 2.0)  # Chose random std_dev between 2 and 30
    max_value = random.uniform(0.5, 50)
    return mean_x, mean_y, std_dev, max_value

folder = "toy_real_data"
save_data = True

if not os.path.exists(f"{project_folder}/{folder}"):
    os.mkdir(f"{project_folder}/{folder}")

if not os.path.exists(f"{project_folder}/{folder}/dirty"):
    os.mkdir(f"{project_folder}/{folder}/dirty")

phase_centers = {}
keys_list = []
sky_sources_info = {}

noise_std = 5e-5
img_multiplier = 1e-5

for i in range(0, 2):
    image = np.zeros((512, 512))

    key = f"{i:06}"  # could be any unique string
    keys_list.append(key)
    sky_sources_info[key] = []

    # generate random RA, DEC
    ra_min = 149.43959
    ra_max = 150.756584

    dec_min = 1.555417
    dec_max = 2.880238

    ra = random.uniform(ra_min, ra_max)
    dec = random.uniform(dec_min, dec_max)

    phase_centers[key] = {"RA": ra, "DEC": dec, }

    # Add three Gaussians with random mean values and std_dev
    for _ in range(np.random.randint(1, 5)):
        mean_x, mean_y, std_dev, max_value = generate_random_params()
        max_value = max_value * img_multiplier
        image_before = image.copy()  # Store the image before adding the Gaussian
        image = add_gaussian(image, mean_x, mean_y, std_dev, max_value)

        std_dev = 0.1 * 2.3548 * std_dev

        flux = np.sum(image - image_before)  # Compute flux as sum of all the pixel values in the Gaussian
        SNR = max_value / noise_std

        beam_maj = 0.89
        beam_min = 0.82
        s_min = std_dev
        s_max = std_dev
        SNR_normalized = SNR * (beam_maj * beam_min) / np.sqrt(
            (beam_maj ** 2 + s_max ** 2) * (beam_min ** 2 + s_min ** 2))

        sky_sources_info[key].append([
            ra,
            dec,
            SNR,
            SNR_normalized,
            flux,
            std_dev,
            std_dev,
        ])

    # Scale the array so that its maximum is around 10**(-5)
    filename = f"{i:06}.npy"

    # Plotting the generated image
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    plt.show()

    # Add Gaussian noise on top
    noise = np.random.normal(0, noise_std, image.shape)
    image += noise

    if save_data:
        np.save(f"{project_folder}/{folder}/dirty/{filename}", image)
    # Plotting the generated image
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    plt.show()

if save_data:
    np.save(f"{project_folder}/{folder}/ra_dec.npy", phase_centers)
    np.save(f"{project_folder}/{folder}//sky_keys.npy", keys_list)
