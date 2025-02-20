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
import cv2
from sklearn.cluster import KMeans


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
    with ThreadPoolExecutor(max_workers=6) as executor:
        # Map des images avec la fonction `remove_label` tout en préservant l'ordre
        results = list(tqdm(executor.map(process_image, breast_images), total=len(breast_images)))

    if save :
        # Conserver les résultats dans un fichier .npy
        np.save('preprocess_images.npy', results)

    return results

# ------------------------------------ ROI ----------------------------------- #

# Fonction pour trouver le label dominant parmi les 100 premiers points du haut de l'image
def dominant_label_top(cluster_labels, white_pixel_coords):
    """
    Trouve le label dominant parmi les 10 premiers pixels du haut de l'image.
    
    Parameters:
        cluster_labels (array): Les labels de chaque pixel blanc.
        white_pixel_coords (array): Coordonnées (y, x) des pixels blancs.
    
    Returns:
        int or None: Le label dominant en fonction du nombre d'occurrences.
    """
    top_indices = np.argsort(white_pixel_coords[:, 0])[:100]  # Prendre les 100 premiers en hauteur
    top_labels = cluster_labels[top_indices]
    
    unique_labels, counts = np.unique(top_labels, return_counts=True)
    
    if len(unique_labels) == 0:
        return None  # Aucun label valide
    
    dominant_label = unique_labels[np.argmax(counts)]
    
    return dominant_label + 1


def get_pectoral_mask(image):
    n, d = image.shape
    # ---------------------------------- Amélioration du contraste --------------------------------- #
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    enhanced_image = np.reshape(enhanced_image, (n,d))

    # ---------------------------------- Filtrer les pixels blancs --------------------------------- #
    # Récupérer les coordonnées et intensités des pixels blancs (seuil passé)
    white_pixel_coords = np.column_stack(np.where(enhanced_image > 180))  # Coordonnées (y, x)

    # ---------------------------------- Appliquer KMeans --------------------------------- #
    # Appliquer KMeans sur les pixels blancs (par exemple, 3 clusters)
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(white_pixel_coords)

    # Labels des clusters pour les pixels blancs
    cluster_labels = kmeans.labels_

    # ---------------------------------- Créer une image des clusters --------------------------------- #
    cluster_image = np.zeros_like(image)  # Image vide (même taille que l'originale)
    cluster_image[white_pixel_coords[:, 0], white_pixel_coords[:, 1]] = cluster_labels + 1  # + 1 pour éviter le 0

    dom_label = dominant_label_top(cluster_labels, white_pixel_coords)

    pectoral_mask = np.where(cluster_image == dom_label, cluster_image, 0)

    return enhanced_image, pectoral_mask

def get_region_of_interest(breast_images):
    images_roi = []
    for i in tqdm(range(len(breast_images)), desc="Removing Pectoral Muscle"):
        image = breast_images[i]

        enhanced_image, pectoral_mask = get_pectoral_mask(image)

        image_roi = enhanced_image.copy()
        image_roi[(pectoral_mask == 2) | (pectoral_mask == 1)] = 0

        images_roi += [image_roi]

    return images_roi