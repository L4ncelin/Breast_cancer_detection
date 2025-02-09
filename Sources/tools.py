from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import pandas as pd
import os
from tqdm import tqdm
from skimage import morphology
from skimage import img_as_ubyte
from concurrent.futures import ThreadPoolExecutor


def load_images():
    # The folder path where images are stored
    folder_path = 'Data/all-mias/'

    # Get all file with .pgm extension
    pgm_files = [folder_path + f for f in os.listdir(folder_path) if f.endswith('.pgm')]

    # Load all images
    breast_images = []
    for i in tqdm(range(len(pgm_files))):
        image = Image.open(pgm_files[i])

        breast_images += [image]

    return breast_images

def load_infos():
    infos_df = pd.read_csv("Data/breasts_infos.txt", sep='\s+', names=["image_idx", "tissue", "abnormality", "severity", "x_coord", "y_coord", "radius"])

    infos_df = infos_df.fillna(0) # Fill NaN
    infos_df["x_coord"] = pd.to_numeric(infos_df["x_coord"], errors="coerce").fillna(0).astype(int) # Convert objetc to int
    infos_df["y_coord"] = pd.to_numeric(infos_df["y_coord"], errors="coerce").fillna(0).astype(int)

    return infos_df


def remove_label(image):    

    thresh = 20  # Seuil entre 0 et 255, ajustez selon vos besoins
    binary_image = image.point(lambda p: 255 if p > thresh else 0)

    image = np.array(image)
    binary_image = np.array(binary_image)

    # Convert to a binary mask (True for 255, False for 0)
    binary_mask = binary_image > 0

    # Apply erosion and dilation
    selem1 = morphology.square(100)  # Structuring element, here a 3x3 square
    eroded_image = morphology.erosion(binary_mask, selem1)
    selem2 = morphology.disk(70)
    dilated_image = morphology.dilation(eroded_image, selem2)
    
    # Convert the binary results back to an image
    dilated_image = img_as_ubyte(dilated_image)

    # Convert dilated image to mask (True for dilated, False for everything else)
    dilated_mask = dilated_image > 0

    # Apply the mask to the original image (use the dilated_mask to select areas)
    masked_image = np.zeros_like(image)
    masked_image[dilated_mask] = image[dilated_mask]

    # Convert the result back to an image (if needed)
    masked_image = img_as_ubyte(masked_image)

    return masked_image

def process_image(image):
    return remove_label(image)

def get_preprocess_images(breast_images:list=[], save:bool=False) -> list:

    # Parallélisation
    with ThreadPoolExecutor() as executor:
        # Map des images avec la fonction `remove_label` tout en préservant l'ordre
        results = list(tqdm(executor.map(process_image, breast_images), total=len(breast_images)))

    if save :
        # Conserver les résultats dans un fichier .npy
        np.save('preprocess_images.npy', results)

    return results