# Standard libraries
import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Image processing libraries
from PIL import Image
import cv2
from skimage import morphology
from skimage import img_as_ubyte
from skimage.feature import graycomatrix
from skimage.filters import unsharp_mask
from scipy.ndimage import median_filter

# Machine learning libraries
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix

# Optimization libraries
import optuna


def load_images() -> list:
    """Load images from the Data folder.

    Returns:
        list: All breast images
    """
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

def load_infos() -> pd.DataFrame:
    """Load all information about mammography, such as abnormality types, etc.

    Returns:
        pd.DataFrame: Information dataframe
    """
    infos_df = pd.read_csv("Data/breasts_infos.txt", sep='\s+', names=["image_idx", "tissue", "abnormality", "severity", "x_coord", "y_coord", "radius"])

    infos_df = infos_df.fillna(0) # Fill NaN
    infos_df["x_coord"] = pd.to_numeric(infos_df["x_coord"], errors="coerce").fillna(0).astype(int) # Convert objetc to int
    infos_df["y_coord"] = pd.to_numeric(infos_df["y_coord"], errors="coerce").fillna(0).astype(int)

    return infos_df


def remove_label(image):
    """Remove the label present in each image to isolate the breast.
    """    
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
    """Apply the removing process on all images, in parallel.

    Args:
        breast_images (list, optional): PIL Images. Defaults to [].
        save (bool, optional): Whether to save the list of images at the end of the function. Defaults to False.

    Returns:
        list: All images with labels removed
    """
    # Parallelisation
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(tqdm(executor.map(process_image, breast_images), total=len(breast_images)))

    if save :
        # Save file in .npy
        np.save('preprocess_images.npy', results)

    return results

# ------------------------------------ ROI ----------------------------------- #

def dominant_label_top(cluster_labels, white_pixel_coords):
    """
    Find the dominant label between the first 10 pixels at the top of the image.
    
    Parameters:
        cluster_labels (array): Labels for white pixels.
        white_pixel_coords (array): Coordinnates (y, x) for white pixels.
    
    Returns:
        int or None: Dominant label.
    """
    top_indices = np.argsort(white_pixel_coords[:, 0])[:100] 
    top_labels = cluster_labels[top_indices]
    
    unique_labels, counts = np.unique(top_labels, return_counts=True)
    
    if len(unique_labels) == 0:
        return None  # No label valid
    
    dominant_label = unique_labels[np.argmax(counts)]
    
    return dominant_label + 1


def get_pectoral_mask(image):
    """Getting the pectoral muscle mask to further remove it and extract the Region Of Interest.

    Args:
        image (PIL): Original image

    Returns:
        PIL image, np.array : Enhanced image, mask of the pectoral region
    """
    n, d = image.shape
    # ---------------------------------- Increase contrast quality --------------------------------- #
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    enhanced_image = np.reshape(enhanced_image, (n,d))

    # ---------------------------------- Filter white pixels --------------------------------- #
    white_pixel_coords = np.column_stack(np.where(enhanced_image > 180))

    # ---------------------------------- Apply KMeans --------------------------------- #
    n_clusters = 2 # 2 cluster that are the breast and the pectoral muscle
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(white_pixel_coords)

    # Labels 
    cluster_labels = kmeans.labels_

    # ---------------------------------- Create an image of clusters --------------------------------- #
    cluster_image = np.zeros_like(image) 
    cluster_image[white_pixel_coords[:, 0], white_pixel_coords[:, 1]] = cluster_labels + 1  # + 1 to pass the 0

    dom_label = dominant_label_top(cluster_labels, white_pixel_coords)

    pectoral_mask = np.where(cluster_image == dom_label, cluster_image, 0)

    return enhanced_image, pectoral_mask

def get_region_of_interest(breast_images) -> list:
    """Getting all images without the pectoral muscle.

    Args:
        breast_images (list): list of original PIL images

    Returns:
        list: All mages without the pectoral mask
    """
    images_roi = []
    for i in tqdm(range(len(breast_images)), desc="Removing Pectoral Muscle"):
        image = breast_images[i]

        enhanced_image, pectoral_mask = get_pectoral_mask(image)

        image_roi = image.copy()
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

        # Extracting GLCM
        P = glcm[:, :, 0, angle]

        # Indices of gray level
        i, j = np.indices(P.shape)

        # Compute features
        autocorrelation = np.sum(i * j * P)
        contrast = np.sum(P * (i - j) ** 2)
        mean = np.sum(i * P) 
        cluster_prominence = np.sum((i + j - 2 * mean) ** 4 * P)
        entropy = -np.sum(P * np.log2(P + 1e-10))

        autocorrelations += [autocorrelation]
        contrasts += [contrast]
        cluster_prominences += [cluster_prominence]
        entropies += [entropy]

    avg_autocorrelation = np.mean(autocorrelations)
    avg_contrast = np.mean(contrasts)
    avg_cluster_prominence = np.mean(cluster_prominences)
    avg_entropy = np.mean(entropies)

    return [avg_autocorrelation, avg_contrast, avg_cluster_prominence, avg_entropy]


def compute_glrlm(image, max_gray_level, angles=[0, 45, 90, 135]) -> dict:
    """Compute GLRLM Matrix.

    Returns:
        dict: GLRLM matrices for each angles
    """
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

        # Extracting GLRLM
        P = glrlm[angle]  # Matrix for one angle

        G, R = P.shape # G gray level, R run length

        total = np.sum(P)

        runs = np.arange(1, R + 1)
        gray_levels = np.arange(1, G + 1)


        sum_over_runs = np.sum(P, axis=1)
        sum_over_gray = np.sum(P, axis=0) 

        # Compute features
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

def get_features_from_images(image_roi:list) -> pd.DataFrame:
    """Compute all 9 features from GLCM and GLRLM on all images.

    Args:
        image_roi (list): All PIL preprocess images

    Returns:
        pd.DataFrame: Dataframe of 9 features for each image
    """
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
        float_flag = False
        # GLCM
        try:
            glcm = graycomatrix(image, distances=[1], angles=[0., 0.78539816, 1.57079633, 2.35619449], levels=256, symmetric=True, normed=False)
        except ValueError:
            float_flag = True
            image_255 = (image * 255).astype(np.uint8)
            glcm = graycomatrix(image_255, distances=[1], angles=[0., 0.78539816, 1.57079633, 2.35619449], levels=256, symmetric=True, normed=False)

        avg_autocorrelation, avg_contrast, avg_cluster_prominence, avg_entropy = compute_features_glcm(glcm)

        autocorrelation += [avg_autocorrelation]
        contrast += [avg_contrast]
        cluster_prominence += [avg_cluster_prominence]
        entropy += [avg_entropy]

        # GLRLM
        if float_flag == True:
            glrlm = compute_glrlm(image_255, max_gray_level=256, angles=[0, 45, 90, 135])
        else:
            glrlm = compute_glrlm(image, max_gray_level=256, angles=[0, 45, 90, 135])
        avg_SRE, avg_LRE, avg_GLNU, avg_SRLGE, avg_LRLGE = compute_features_glrlm(glrlm)

        SRE += [avg_SRE]
        LRE += [avg_LRE]
        GLNU += [avg_GLNU]
        SRLGE += [avg_SRLGE]
        LRLGE += [avg_LRLGE]

    data = pd.DataFrame({"autocorrelation":autocorrelation, "contrast":contrast, "cluster_prominence":cluster_prominence, "entropy":entropy, "SRE":SRE, "LRE":LRE, "GLNU":GLNU, "SRLGE":SRLGE, "LRLGE":LRLGE})

    return data

# ---------------------------------- Filter ---------------------------------- #

def apply_clahe_filter_on_all_images(images_roi:list) -> list:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) filter on all images.

    Args:
        images_roi (list): All original PIL images

    Returns:
        list: All images with CLAHE Filter applied
    """
    filtered_images = []

    for image in tqdm(images_roi, desc="Applying CLAHE filter..."):
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
        filtered_image = clahe.apply(image)

        filtered_images += [filtered_image]

    return filtered_images

def apply_MF_and_CLAHE_and_USM_on_all_images(images_roi:list) -> list:
    """Apply the combination of Median Filter, CLAHE filter and UnSharp Maksing on all images.

    Args:
        images_roi (list): All original PIL images

    Returns:
        list: All images with MF&CLAHE&USM Filter applied
    """
    filtered_images = []

    for image in tqdm(images_roi, desc="Applying MF & CLAHE & USM..."):
        # MF
        mf_image = median_filter(image, size=5)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(mf_image)

        # USM
        filtered_image = unsharp_mask(clahe_image, radius=2, amount=2)

        filtered_images += [filtered_image]

    return filtered_images

# ----------------------------------- Model ---------------------------------- #

def compute_classification_metrics(y_true, y_pred, verbose:bool=True) -> dict:
    """Compute the performance metrics for a prediction and its true value.

    Args:
        y_true (pd.Series): Real value
        y_pred (pd.Series): Prediction from the model
        verbose (bool, optional): Print results or not. Defaults to True.

    Returns:
        dict: _description_
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    ppv = precision_score(y_true, y_pred)
    npv = tn / (tn + fn)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    balanced_accuracy = (sensitivity + specificity) / 2

    if verbose:
        # Prin Results
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Sensitivity (Recall): {sensitivity:.3f}")
        print(f"Specificity: {specificity:.3f}")
        print(f"PPV (Precision): {ppv:.3f}")
        print(f"NPV: {npv:.3f}")
        print(f"F1-score: {f1:.3f}")
        print(f"AUC: {auc:.3f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.3f}")

    return {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "F1-score": f1,
        "AUC": auc,
        "Balanced Accuracy": balanced_accuracy
    }


def objective(trial, X, y):
    """Objective function for Optuna optimization on Random Forest.

    Args:
        trial : One trial of a model
        X (pd.DataFrame): Features data
        y (pd.Series): Target value

    Returns:
        float: Accuracy score of the given model 
    """
    n_estimators = trial.suggest_int("n_estimators", 10, 200, step=10)
    max_depth = trial.suggest_int("max_depth", 2, 20, step=1)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(
        random_state=42,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def get_best_hyperparameters_optuna(X, y, n_trials:int=20) -> dict:
    """Create an Optuna study to find the best hyperparameters of a given model.

    Args:
        X (pd.DataFrame): Features data
        y (pd.Series): Target value
        n_trials (int, optional): Number of trial to try. Defaults to 20.

    Returns:
        dict: Best hyperparameters
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print(f"\nMeilleurs hyperparamètres trouvés : {study.best_params}")
    print(f"Meilleure accuracy (LOO) : {study.best_value:.4f}")

    return study.best_params