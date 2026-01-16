# PID_implementation
import os
import cv2
import numpy as np
import pandas as pd

from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize

import tensorflow as tf
from tensorflow import keras
from patchify import patchify, unpatchify

import matplotlib.pyplot as plt

# Rebuild Model Architecture due to TF compatibility issues

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    """Your exact U-Net architecture"""
    inputs = keras.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = keras.layers.Dropout(0.1)(c1)
    c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    
    c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = keras.layers.Dropout(0.1)(c2)
    c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
     
    c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = keras.layers.Dropout(0.2)(c3)
    c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
     
    c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = keras.layers.Dropout(0.2)(c4)
    c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = keras.layers.Dropout(0.3)(c5)
    c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # Expansive path 
    u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = keras.layers.Dropout(0.2)(c6)
    c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = keras.layers.Dropout(0.2)(c7)
    c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = keras.layers.Dropout(0.1)(c8)
    c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = keras.layers.concatenate([u9, c1], axis=3)
    c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = keras.layers.Dropout(0.1)(c9)
    c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model

class PetriDishMapper:
    
    """
    Maps cropped petri-dish pixel coordinates to simulation coordinates
    """

    def __init__(self, pixel_size):
        self.PIXEL_SIZE = pixel_size
        self.SIM_TOP_LEFT_X = 0.10775
        self.SIM_TOP_LEFT_Y = 0.062  # 0.088 - 0.026
        self.SIM_BOTTOM_RIGHT_X = 0.25775
        self.SIM_BOTTOM_RIGHT_Y = 0.212  # 0.238 - 0.026
        self.SIM_WIDTH = self.SIM_BOTTOM_RIGHT_X - self.SIM_TOP_LEFT_X
        self.SIM_HEIGHT = self.SIM_BOTTOM_RIGHT_Y - self.SIM_TOP_LEFT_Y
        self.DISPENSE_HEIGHT = 0.175

        
    def pixel_to_sim(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates to simulation coordinates.
        Accounts for 90-degree clockwise rotation of petri dish in simulation,
        then applies a 180-degree flip.
        """

        norm_x = pixel_x / self.PIXEL_SIZE
        norm_y = 1.0 - (pixel_y / self.PIXEL_SIZE)

        # Existing 90° clockwise rotation (UNCHANGED)
        sim_x = self.SIM_TOP_LEFT_X + norm_y * self.SIM_WIDTH
        sim_y = self.SIM_TOP_LEFT_Y + (1.0 - norm_x) * self.SIM_HEIGHT

        # ---- 180° ROTATION (NO CENTER VARIABLES) ----
        sim_x = self.SIM_TOP_LEFT_X + self.SIM_BOTTOM_RIGHT_X - sim_x
        sim_y = self.SIM_TOP_LEFT_Y + self.SIM_BOTTOM_RIGHT_Y - sim_y

        sim_z = self.DISPENSE_HEIGHT

        return sim_x, sim_y, sim_z


    
    def validate_coords(self, sim_x, sim_y, sim_z):

        """
        Check if coordinates are within bounds
        """
        
        x_valid = self.SIM_TOP_LEFT_X <= sim_x <= self.SIM_BOTTOM_RIGHT_X
        y_valid = self.SIM_TOP_LEFT_Y <= sim_y <= self.SIM_BOTTOM_RIGHT_Y
        z_valid = 0.17 <= sim_z <= 0.20
        return x_valid and y_valid and z_valid
    
def f1(y_true, y_pred):

    """
    F1 metric
    """

    def recall_m(y_true, y_pred):
        TP = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        Positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
        recall = TP / (Positives + tf.keras.backend.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        Pred_Positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + tf.keras.backend.epsilon())
        return precision
    
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

def padder(image, patch_size):

    """
    Adds padding to an image to make its dimensions divisible by a specified patch size.

    This function calculates the amount of padding needed for both the height and width of an image so that its dimensions become divisible by the given patch size. The padding is applied evenly to both sides of each dimension (top and bottom for height, left and right for width). If the padding amount is odd, one extra pixel is added to the bottom or right side. The padding color is set to black (0, 0, 0).

    Parameters:
    - image (numpy.ndarray): The input image as a NumPy array. Expected shape is (height, width, channels).
    - patch_size (int): The patch size to which the image dimensions should be divisible. It's applied to both height and width.

    Returns:
    - numpy.ndarray: The padded image as a NumPy array with the same number of channels as the input. Its dimensions are adjusted to be divisible by the specified patch size.

    Example:
    - padded_image = padder(cv2.imread('example.jpg'), 128)

    """

    h, w = image.shape[:2]

    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding  = ((w // patch_size) + 1) * patch_size - w

    top  = height_padding // 2
    bottom = height_padding - top

    left  = width_padding // 2
    right = width_padding - left

    # Ensure non-negative
    top = max(top, 0)
    bottom = max(bottom, 0)
    left = max(left, 0)
    right = max(right, 0)

    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=(0,0,0)
    )

    return padded, (top, bottom, left, right)

def crop_petri_dish(image):

    """
    Input: Takes list of images
    Returns: List of cropped images
    """

    original = image.copy()

    # strong blur
    blurred = cv2.GaussianBlur(image, (101,101), 0)

    # edges
    edges = cv2.Canny(blurred, 0, 15)

    # close gaps
    kernel = np.ones((15,15), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # contours
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(contours, key=cv2.contourArea)

    # bounding rectangle
    x,y,w,h = cv2.boundingRect(big)
    crop_coords = (x,y,w,h)

    cropped = original[y:y+h, x:x+w]

    return cropped, crop_coords
    
def predict_full_image(model, padded_img, patch_size=128):

    """
    Predicts segmentation mask for full image using patch-based approach.
    
    Parameters:
    - model: Trained segmentation model
    - padded_img: Padded input image
    - patch_size: Size of patches for prediction
    
    Returns:
    - reconstructed: Full-size predicted mask
    """

    patches = patchify(padded_img, (patch_size, patch_size, 3), step=patch_size)

    i, j = patches.shape[:2]
    patches_reshaped = patches.reshape(-1, patch_size, patch_size, 3)

    preds = model.predict(patches_reshaped)

    preds = preds.reshape(i, j, patch_size, patch_size)
    reconstructed = unpatchify(preds, padded_img.shape[:2])

    return reconstructed

def unpad(image, pad_vals):

    """
    Unpads image using values stored in padding function
    """

    top, bottom, left, right = pad_vals
    return image[top: image.shape[0]-bottom, left: image.shape[1]-right]

def uncrop(full_shape, crop_coords, mask_crop):

    """
    Uncrops image using values stored in cropping function
    """

    H, W = full_shape[:2]
    x, y, w, h = crop_coords

    output = np.zeros((H, W), dtype=mask_crop.dtype)
    output[y:y+h, x:x+w] = mask_crop
    return output

def run_pipeline_on_image(img_path, model, patch_size=128):

    """
    Full pipeline
    Inputs: Images, model, patch size
    Outputs: Patched, uncropped, unpadded, predicted mask
    """

    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Crop dish
    cropped, crop_coords = crop_petri_dish(img_gray)
    
    # Store original dimensions
    original_shape = img_gray.shape
    cropped_shape = cropped.shape
    
    # Convert to 3-channel for model
    cropped_3ch = np.stack([cropped, cropped, cropped], axis=-1)
    
    # Pad
    padded, pad_vals = padder(cropped_3ch, patch_size)
    
    # Predict
    pred_mask = predict_full_image(model, padded, patch_size)
    
    # Remove padding
    pred_mask_unpadded = unpad(pred_mask, pad_vals)
    
    # Uncrop back to original resolution
    final_mask = uncrop(img_gray.shape, crop_coords, pred_mask_unpadded)
    
    # Return metadata
    metadata = {
        'original_shape': original_shape,
        'cropped_shape': cropped_shape,
        'crop_coords': crop_coords  # (x, y, w, h)
    }
    
    return img_gray, final_mask, metadata

def final_cleaning_pipeline(pred, y_min=380, y_max=2700, 
                           closing_kernel=3, min_size=100, isolation_threshold=50):
    """
    Complete cleaning pipeline - NO X-band filtering here!
    Input: Predicted masked image from model
    Output: Cleaned mask ready for segmentation
    """
    from scipy.ndimage import distance_transform_edt
    
    # Convert to uint8 if needed
    img = (pred * 255).astype(np.uint8) if pred.max() <= 1 else pred.astype(np.uint8)
    
    # Step 1: Crop Y-range ONLY
    img[0:y_min, :] = 0
    img[y_max:, :] = 0
    
    # NO X-RANGE CROPPING HERE!
    
    # Step 2: CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
    enhanced = clahe.apply(img)
    
    # Step 3: Otsu threshold
    _, bw = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 4: Closing to reconnect roots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    
    # Step 5: Size filter
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    
    size_filtered = np.zeros_like(bw)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            size_filtered[labels == i] = 255
    
    # Step 6: Remove isolated noise
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(size_filtered, connectivity=8)
    
    cleaned = np.zeros_like(bw)
    
    for i in range(1, num_labels2):
        area = stats2[i, cv2.CC_STAT_AREA]
        component_mask = (labels2 == i)
        
        # For small components, check if they're isolated
        if area < 500:
            # Distance to other components
            other_components = size_filtered.copy()
            other_components[component_mask] = 0
            
            if np.any(other_components):
                dist = distance_transform_edt(~other_components.astype(bool))
                min_dist_to_others = dist[component_mask].min()
                
                # Keep if close to other structures
                if min_dist_to_others <= isolation_threshold:
                    cleaned[component_mask] = 255
            else:
                cleaned[component_mask] = 255
        else:
            # Keep all large components
            cleaned[component_mask] = 255
    
    return cleaned

def segment_plants_by_min_distance(cleaned_mask, distance_threshold=80):

    """
    Segments plants by merging components whose pixel-to-pixel
    distance is below a threshold (in any direction).
    """

    from scipy.spatial.distance import cdist
    
    # Step 1: initial components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, 8)
    if num_labels <= 1:
        return 0, labels

    # Extract pixel coords of each component
    component_pixels = {}
    for comp_id in range(1, num_labels):
        ys, xs = np.where(labels == comp_id)
        coords = np.column_stack((ys, xs))
        component_pixels[comp_id] = coords

    # Union-Find structure
    parent = {cid: cid for cid in range(1, num_labels)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Step 2: compute pairwise minimal distances
    for i in range(1, num_labels):
        for j in range(i + 1, num_labels):

            pts_i = component_pixels[i]
            pts_j = component_pixels[j]

            # Compute Euclidean distances between pixel sets
            dist = cdist(pts_i, pts_j).min()

            if dist < distance_threshold:
                union(i, j)

    # Step 3: build final plant label map
    final_labels = np.zeros_like(cleaned_mask)
    plant_id_map = {}
    plant_count = 0

    for comp_id in range(1, num_labels):
        root = find(comp_id)
        if root not in plant_id_map:
            plant_count += 1
            plant_id_map[root] = plant_count
        final_labels[labels == comp_id] = plant_id_map[root]

    return plant_count, final_labels

def compute_primary_root_length(mask):
    """
    Takes Segmented plants, Skeletonizes the main root, 
    Outputs maximum of 5 respective lengths.
    """
    if mask.sum() == 0:
        return 0

    skeleton = skeletonize(mask > 0).astype(np.uint8)

    ys, xs = np.where(skeleton == 1)
    coords = np.column_stack((ys, xs))

    if len(coords) < 5:
        return 0

    dist_matrix = cdist(coords, coords)
    max_idx = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
    length = dist_matrix[max_idx]

    return float(length)

def compute_root_tip_coordinate(mask):
    """
    Get bottom-most point (root tip) instead of length.
    Replaces compute_primary_root_length for coordinate extraction.
    """
    if mask.sum() == 0:
        return None
    
    ys, xs = np.where(mask > 0)
    
    if len(ys) == 0:
        return None
    
    # Find BOTTOM of root (maximum Y = deepest point)
    max_y_idx = np.argmax(ys)
    tip_y = ys[max_y_idx]
    tip_x = xs[max_y_idx]
    
    return int(tip_x), int(tip_y)

def process_image_with_y_band(img_path, model, min_size=100, isolation_threshold=50, 
                              distance_threshold=80, max_plants=5, 
                              y_min_validation=400, y_max_validation=800,  # For plant validation
                              x_min_validation=1000, x_max_validation=3300,  # For plant validation
                              y_min_pixels=380, y_max_pixels=2700):  # For pixel removal
    """
    Process image with Y-band and optional X-band filtering.
    
    Parameters:
    - y_min_validation, y_max_validation: Y-range for validating plant START positions
    - x_min_validation, x_max_validation: X-range for validating plant START positions  
    - y_min_pixels, y_max_pixels: Y-range for removing pixels (wider range)
    """

    _, pred, metadata = run_pipeline_on_image(img_path, model)
    
    # Use WIDE Y-range for pixel removal
    cleaned = final_cleaning_pipeline(
        pred, 
        y_min=y_min_pixels,  # 380 - wider!
        y_max=y_max_pixels,  # 2700 - wider!
        min_size=min_size, 
        isolation_threshold=isolation_threshold
    )
    
    plant_count, plant_mask = segment_plants_by_min_distance(cleaned, distance_threshold=distance_threshold)

    # Y-band VALIDATION: Check if TOP of plant is in valid Y-range
    if y_min_validation is not None and y_max_validation is not None:
        for pid in range(1, plant_count+1):
            ys, xs = np.where(plant_mask == pid)
            if len(ys) == 0:
                continue
            y_top = ys.min()
            if not (y_min_validation <= y_top <= y_max_validation):
                plant_mask[plant_mask == pid] = 0
        
        # Reindex
        unique_ids = np.unique(plant_mask)
        unique_ids = unique_ids[unique_ids != 0]
        new_mask = np.zeros_like(plant_mask)
        for idx, uid in enumerate(unique_ids):
            new_mask[plant_mask == uid] = idx + 1
        plant_mask = new_mask
        plant_count = len(unique_ids)

    # X-band VALIDATION: Check if TOP of plant is in valid X-range
    if x_min_validation is not None and x_max_validation is not None:
        for pid in range(1, plant_count+1):
            ys, xs = np.where(plant_mask == pid)
            if len(ys) == 0:
                continue
            
            # Get the TOP point (minimum Y) and its X-coordinate
            top_idx = np.argmin(ys)
            x_top = xs[top_idx]
            
            # Remove plant if its top is outside X-range
            if not (x_min_validation <= x_top <= x_max_validation):
                plant_mask[plant_mask == pid] = 0
        
        # Reindex again
        unique_ids = np.unique(plant_mask)
        unique_ids = unique_ids[unique_ids != 0]
        new_mask = np.zeros_like(plant_mask)
        for idx, uid in enumerate(unique_ids):
            new_mask[plant_mask == uid] = idx + 1
        plant_mask = new_mask
        plant_count = len(unique_ids)

    # Remove smallest plants if too many
    if plant_count > max_plants:
        areas = [(pid, np.sum(plant_mask == pid)) for pid in range(1, plant_count+1)]
        areas_sorted = sorted(areas, key=lambda x: x[1])
        for pid, _ in areas_sorted[:plant_count - max_plants]:
            plant_mask[plant_mask == pid] = 0
        
        # Reindex again
        unique_ids = np.unique(plant_mask)
        unique_ids = unique_ids[unique_ids != 0]
        new_mask = np.zeros_like(plant_mask)
        for idx, uid in enumerate(unique_ids):
            new_mask[plant_mask == uid] = idx + 1
        plant_mask = new_mask
        plant_count = len(unique_ids)

    # Compute lengths
    lengths = []
    for pid in range(1, plant_count+1):
        mask = (plant_mask == pid).astype(np.uint8)
        length = compute_primary_root_length(mask)
        lengths.append(length)

    return plant_mask, lengths, metadata

def learn_plant_positions(image_folder, model, sample_size=None):

    """
    Learn typical x-positions of plants from images with 5 detected plants.
    
    Parameters:
    - image_folder: Path to folder containing training images
    - model: Trained segmentation model
    - sample_size: Number of images to analyze (None = use all)
    
    Returns:
    - avg_positions: List of 5 average x-coordinates for plant positions
    """

    position_centroids = [[] for _ in range(5)]
    
    image_files = sorted(os.listdir(image_folder))
    if sample_size:
        image_files = image_files[:sample_size]
    
    print(f"Analyzing {len(image_files)} images to learn plant positions...")
    
    for img_name in image_files:
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg", ".tif")):
            continue
        
        img_path = os.path.join(image_folder, img_name)
        plant_mask, lengths, metadata = process_image_with_y_band(img_path, model)
        
        # Only use images with exactly 5 plants for training
        unique_plants = np.unique(plant_mask)
        unique_plants = unique_plants[unique_plants > 0]
        
        if len(unique_plants) == 5:
            # Get centroids for each plant
            centroids = []
            for pid in unique_plants:
                ys, xs = np.where(plant_mask == pid)
                centroid_x = xs.mean()
                centroids.append(centroid_x)
            
            # Sort by x-position (left to right)
            centroids.sort()
            
            # Store x-positions for each position
            for i, cx in enumerate(centroids):
                position_centroids[i].append(cx)
    
    # Calculate average positions
    if all(len(pos) > 0 for pos in position_centroids):
        avg_positions = [np.mean(positions) for positions in position_centroids]
        print(f"✓ Learned positions from {len(position_centroids[0])} images")
        print(f"  Average X-positions: {[f'{x:.1f}' for x in avg_positions]}")
    else:
        # Fallback: equal spacing if not enough training data
        print("⚠️ Not enough training data. Using equal spacing fallback.")
        
        # Get image width from first image
        first_img = None
        for img_name in image_files:
            if img_name.lower().endswith((".jpg", ".png", ".jpeg", ".tif")):
                img_path = os.path.join(image_folder, img_name)
                first_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                break
        
        if first_img is not None:
            width = first_img.shape[1]
        else:
            width = 4000  # Default fallback width
        
        avg_positions = [width * (i + 1) / 6 for i in range(5)]
        print(f"  Using equal spacing: {[f'{x:.1f}' for x in avg_positions]}")
    
    return avg_positions

def create_zone_boundaries(avg_positions):

    """
    Create zone boundaries as midpoints between average positions.
    Zone i covers plants that should be in position i.
    
    Parameters:
    - avg_positions: List of 5 average x-coordinates
    
    Returns:
    - boundaries: List of 6 boundary values defining 5 zones
    """

    boundaries = [0]  # Start at x=0
    
    # Midpoints between consecutive positions
    for i in range(len(avg_positions) - 1):
        midpoint = (avg_positions[i] + avg_positions[i + 1]) / 2
        boundaries.append(midpoint)
    
    boundaries.append(float('inf'))  # End at x=infinity
    
    return boundaries

def assign_plant_to_position(centroid_x, zone_boundaries):

    """
    Assign a plant to position 0-4 based on its x-centroid.
    
    Parameters:
    - centroid_x: X-coordinate of plant centroid
    - zone_boundaries: List of zone boundary values
    
    Returns:
    - position: Integer 0-4 representing plant position (left to right)
    """

    for i in range(len(zone_boundaries) - 1):
        if zone_boundaries[i] <= centroid_x < zone_boundaries[i + 1]:
            return i
    
    return len(zone_boundaries) - 2  # Fallback to last position

def process_image_with_positions(img_path, model, zone_boundaries, 
                                 min_size=100, isolation_threshold=50,
                                 distance_threshold=80, 
                                 y_min_validation=400, y_max_validation=800,
                                 return_coords=False):
    """
    Process image and assign detected plants to their 5 positions (0-4).
    NEW: Can return either lengths OR coordinates based on return_coords flag.
    """
    
    # Get detections WITH metadata
    plant_mask, lengths, metadata = process_image_with_y_band(
        img_path, model, 
        min_size=min_size, 
        isolation_threshold=isolation_threshold,
        distance_threshold=distance_threshold, 
        max_plants=5, 
        y_min_validation=y_min_validation,
        y_max_validation=y_max_validation
    )
    
    # Get crop offset
    crop_x, crop_y, crop_w, crop_h = metadata['crop_coords']
    
    # Initialize output
    if return_coords:
        position_data = [None] * 5
    else:
        position_data = [0.0] * 5
    
    unique_plants = np.unique(plant_mask)
    unique_plants = unique_plants[unique_plants > 0]
    
    for pid in unique_plants:
        ys, xs = np.where(plant_mask == pid)
        centroid_x = xs.mean()
        position = assign_plant_to_position(centroid_x, zone_boundaries)
        
        # CREATE MASK FIRST (before if/else)
        mask = (plant_mask == pid).astype(np.uint8)
        
        if return_coords:
            # Get root tip in original image coordinates
            data_full = compute_root_tip_coordinate(mask)
            
            if data_full is not None:
                x_full, y_full = data_full
            
                # Convert FULL IMAGE → CROPPED DISH coordinates
                x_crop = x_full - crop_x
                y_crop = y_full - crop_y
            
                data = (int(x_crop), int(y_crop))
            else:
                data = None
                
            if position_data[position] is None:
                position_data[position] = data
            elif data is not None:
                if data[1] > position_data[position][1]:
                    position_data[position] = data
    
    return plant_mask, position_data, metadata

def process_dataset_with_positions(image_folder, model, csv_path, 
                                   learning_sample_size=None):

    """
    Process all images and output exactly 5 plant measurements per image.
    
    Parameters:
    - image_folder: Path to folder containing test images
    - model: Trained segmentation model
    - csv_path: Output path for CSV file
    - learning_sample_size: Number of images to use for learning positions (None = all)
    
    Returns:
    - df: DataFrame with results
    - plant_masks_dict: Dictionary of plant segmentation masks
    
    CSV format:
        Plant ID, Length (px)
    Where Plant ID is imagename_position (e.g., test_image_01.png_1)
    """

    # Step 1: Learn plant positions from images
    print("=" * 60)
    print("STEP 1: Learning plant positions")
    print("=" * 60)
    avg_positions = learn_plant_positions(image_folder, model, learning_sample_size)
    zone_boundaries = create_zone_boundaries(avg_positions)
    
    print(f"\nZone boundaries: {[f'{b:.1f}' for b in zone_boundaries]}")
    
    # Step 2: Process all images with learned positions
    print("\n" + "=" * 60)
    print("STEP 2: Processing all images")
    print("=" * 60)
    
    all_entries = []
    plant_masks_dict = {}
    image_files = sorted(os.listdir(image_folder))
    
    for idx, img_name in enumerate(image_files, 1):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg", ".tif")):
            continue
        
        img_path = os.path.join(image_folder, img_name)
        
        # Get lengths for all 5 positions
        plant_mask, position_lengths = process_image_with_positions(
            img_path, model, zone_boundaries
        )
        
        plant_masks_dict[img_name] = plant_mask
        
        # Add 5 rows for this image (one per plant position)
        for plant_id, length in enumerate(position_lengths, start=1):
            all_entries.append({
                "Plant ID": f"{img_name[:-4]}_plant_{plant_id}",  # Changed: unique ID
                "Length (px)": length
            })
        
        detected_count = sum(1 for l in position_lengths if l > 0)
        print(f"[{idx}/{len([f for f in image_files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif'))])}] {img_name}: {detected_count}/5 plants detected")
    
    # Save to CSV
    df = pd.DataFrame(all_entries)
    df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Results saved to {csv_path}")
    print(f"  Total images: {len(plant_masks_dict)}")
    print(f"  Total rows: {len(df)} (5 per image)")
    
    return df, plant_masks_dict

def visualize_root_tips(img_path, plant_mask, position_coords, metadata, position_names=None):

    """
    Visualize detected root tips on original image.
    
    Parameters:
    - img_path: Path to image
    - plant_mask: Segmentation mask
    - position_coords: List of 5 (pixel_x, pixel_y) or None
    - position_names: Optional list of position labels
    """

    from matplotlib.patches import Circle

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    crop_x, crop_y, _, _ = metadata['crop_coords']

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # LEFT: original image
    axes[0].imshow(img_rgb)
    axes[0].imshow(plant_mask, alpha=0.3, cmap='nipy_spectral')

    detected_count = 0
    for pos_idx, coords in enumerate(position_coords, 1):
        if coords is not None:
            detected_count += 1

            # CROPPED → FULL IMAGE
            fx = coords[0] + crop_x
            fy = coords[1] + crop_y

            circle = Circle((fx, fy), radius=50,
                            color='red', fill=False, linewidth=3)
            axes[0].add_patch(circle)

            label = f"Pos {pos_idx}"
            axes[0].text(fx, fy - 80, label,
                          color='red', fontsize=12, weight='bold',
                          ha='center',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[0].set_title(f"{os.path.basename(img_path)} — {detected_count}/5 plants")
    axes[0].axis('off')
    
    # Right: Simulation coordinate space
    pixel_size = metadata['cropped_shape'][0]
    mapper = PetriDishMapper(pixel_size)
    axes[1].set_xlim(0.10, 0.26)
    axes[1].set_ylim(0.06, 0.21)
    axes[1].set_aspect('equal')
    
    # Draw petri dish boundary
    axes[1].add_patch(plt.Rectangle(
        (mapper.SIM_TOP_LEFT_X, mapper.SIM_TOP_LEFT_Y),
        mapper.SIM_WIDTH, mapper.SIM_HEIGHT,
        fill=False, edgecolor='blue', linewidth=2, label='Petri Dish'
    ))
    
    # Plot transformed coordinates
    for pos_idx, coords in enumerate(position_coords, 1):
        if coords is not None:
            sim_x, sim_y, sim_z = mapper.pixel_to_sim(coords[0], coords[1])
            axes[1].plot(sim_x, sim_y, 'ro', markersize=12)
            axes[1].text(sim_x, sim_y + 0.008, 
                        f"{pos_idx}", color='red', fontsize=10, weight='bold', ha='center')
    
    axes[1].set_xlabel('Simulation X (m)', fontsize=12)
    axes[1].set_ylabel('Simulation Y (m)', fontsize=12)
    axes[1].set_title('Simulation Space (Left to Right Order)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()