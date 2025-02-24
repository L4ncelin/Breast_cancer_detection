from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import pandas as pd
import os

from tqdm import tqdm
from skimage import morphology
from skimage import img_as_ubyte
from skimage.feature import graycomatrix

from concurrent.futures import ThreadPoolExecutor
import cv2
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix


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


# --------------------------- Features Exctraction --------------------------- #

def compute_features_glcm(glcm, angles_nb:int=4) -> list:
    """Compute GLCM features (Autocorrelation, contrast, cluster prominence, entopy).

    Returns:
        list: Average feature for all angles computed in GLCM
    """
    autocorrelations = []
    contrasts = []
    cluster_prominences = []
    entropies = []

    for angle in range(angles_nb):

        # Extraction de la matrice GLCM normalisée
        P = glcm[:, :, 0, angle]  # Prend la matrice pour l'angle donné

        # Indices des niveaux de gris
        i, j = np.indices(P.shape)

        # Calcul des features
        autocorrelation = np.sum(i * j * P)
        contrast = np.sum(P * (i - j) ** 2)
        mean = np.sum(i * P)  # Moyenne des intensités
        cluster_prominence = np.sum((i + j - 2 * mean) ** 4 * P)
        entropy = -np.sum(P * np.log2(P + 1e-10))  # Ajout de 1e-10 pour éviter log(0)

        autocorrelations += [autocorrelation]
        contrasts += [contrast]
        cluster_prominences += [cluster_prominence]
        entropies += [entropy]

    avg_autocorrelation = np.mean(autocorrelations)
    avg_contrast = np.mean(contrasts)
    avg_cluster_prominence = np.mean(cluster_prominences)
    avg_entropy = np.mean(entropies)

    return [avg_autocorrelation, avg_contrast, avg_cluster_prominence, avg_entropy]


def compute_glrlm(image, max_gray_level, angles=[0, 45, 90, 135]):

    def get_runs(img, angle):
        if angle == 0:  # horizontal
            lines = img
        elif angle == 90:  # vertical
            lines = img.T  
        elif angle == 45: 
            lines = [np.diag(img, k) for k in range(-img.shape[0]+1, img.shape[1])]
        elif angle == 135: 
            flipped_img = np.fliplr(img) 
            lines = [np.diag(flipped_img, k) for k in range(-flipped_img.shape[0]+1, flipped_img.shape[1])]

        runs = []
        for line in lines:
            if len(line) == 0:
                continue
            run_length = 1
            for i in range(1, len(line)):
                if line[i] == line[i - 1]:
                    run_length += 1
                else:
                    runs.append((line[i - 1], run_length))
                    run_length = 1
            runs.append((line[-1], run_length))  # last run
        return runs

    # Initialisation GLRLMs
    glrlms = {angle: np.zeros((max_gray_level, image.shape[1]), dtype=int) for angle in angles}

    # Compute fo each angle
    for angle in angles:
        runs = get_runs(image, angle)
        for gray_level, run_length in runs:
            if run_length <= image.shape[1]:
                glrlms[angle][gray_level, run_length - 1] += 1

    return glrlms

def compute_features_glrlm(glrlm, angles:list=[0,45,90,135]) -> list:
    """Compute GLRLM features (Short run emphasis, Long run emphasis, Gray-level non-uniformity, Short run low gray-levvel emphasis, Logn run low gray_level emphasis).

    Returns:
        list: Average feature for all angles computed in GLRLM
    """
    all_SRE = []
    all_LRE = []
    all_GLNU = []
    all_SRLGE = []
    all_LRLGE = []

    for angle in angles:

        # Extraction de la matrice GLRLM
        P = glrlm[angle]  # Prend la matrice pour l'angle donné

        G, R = P.shape # G niveaux de gris, R longueurs des runs

        # Somme totale des valeurs dans la matrice GLRLM
        total = np.sum(P)

        # Indices des niveaux de gris et longueurs des runs
        runs = np.arange(1, R + 1)  # Longueurs de runs (1,2,3,...)
        gray_levels = np.arange(1, G + 1)  # Niveaux de gris (1,2,3,...)

        # Somme des valeurs sur les longueurs des runs et niveaux de gris
        sum_over_runs = np.sum(P, axis=1)  # Somme pour chaque niveau de gris
        sum_over_gray = np.sum(P, axis=0)  # Somme pour chaque longueur de run

        # Calcul des features
        SRE = np.sum(sum_over_gray / (runs ** 2)) / total  # Short Run Emphasis
        LRE = np.sum(sum_over_gray * (runs ** 2)) / total  # Long Run Emphasis
        GLNU = np.sum(sum_over_runs ** 2) / total  # Gray-Level Non-Uniformity
        SRLGE = np.sum(P.T / ((runs[:, None] ** 2) * (gray_levels[None, :] ** 2))) / total  # Short Run Low Gray-Level Emphasis
        LRLGE = np.sum(P.T * (runs[:, None] ** 2) / (gray_levels[None, :]**2)) / total  # Long Run Low Gray-Level Emphasis

        all_SRE += [SRE]
        all_LRE += [LRE]
        all_GLNU += [GLNU]
        all_SRLGE += [SRLGE]
        all_LRLGE += [LRLGE]

    avg_SRE = np.mean(all_SRE)
    avg_LRE = np.mean(all_LRE)
    avg_GLNU = np.mean(all_GLNU)
    avg_SRLGE = np.mean(all_SRLGE)
    avg_LRLGE = np.mean(all_LRLGE)

    return [avg_SRE, avg_LRE, avg_GLNU, avg_SRLGE, avg_LRLGE]

def get_features_from_images(image_roi):
    autocorrelation = []
    contrast = []
    cluster_prominence = []
    entropy = []

    SRE = []
    LRE = []
    GLNU = []
    SRLGE = []
    LRLGE = []

    for image in tqdm(image_roi, desc="Extracting features ..."):
        # GLCM
        glcm = graycomatrix(image, distances=[1], angles=[0., 0.78539816, 1.57079633, 2.35619449], levels=256, symmetric=True, normed=False)
        avg_autocorrelation, avg_contrast, avg_cluster_prominence, avg_entropy = compute_features_glcm(glcm)

        autocorrelation += [avg_autocorrelation]
        contrast += [avg_contrast]
        cluster_prominence += [avg_cluster_prominence]
        entropy += [avg_entropy]

        # GLRLM
        glrlm = compute_glrlm(image, max_gray_level=256, angles=[0, 45, 90, 135])
        avg_SRE, avg_LRE, avg_GLNU, avg_SRLGE, avg_LRLGE = compute_features_glrlm(glrlm)

        SRE += [avg_SRE]
        LRE += [avg_LRE]
        GLNU += [avg_GLNU]
        SRLGE += [avg_SRLGE]
        LRLGE += [avg_LRLGE]

    data = pd.DataFrame({"autocorrelation":autocorrelation, "contrast":contrast, "cluster_prominence":cluster_prominence, "entropy":entropy, "SRE":SRE, "LRE":LRE, "GLNU":GLNU, "SRLGE":SRLGE, "LRLGE":LRLGE})

    return data


# ----------------------------------- Model ---------------------------------- #

def compute_classification_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    ppv = precision_score(y_true, y_pred)
    npv = tn / (tn + fn)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    balanced_accuracy = (sensitivity + specificity) / 2

    # Affichage des résultats
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"PPV (Precision): {ppv:.2f}")
    print(f"NPV: {npv:.2f}")
    print(f"F1-score: {f1:.2f}")
    print(f"AUC: {auc:.2f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2f}")

    return {
        "Accuracy": accuracy,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "PPV (Precision)": ppv,
        "NPV": npv,
        "F1-score": f1,
        "AUC": auc,
        "Balanced Accuracy": balanced_accuracy
    }