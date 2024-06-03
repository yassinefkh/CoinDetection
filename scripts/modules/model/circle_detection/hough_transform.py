import numpy as np

from modules.model.circle_detection.utils import remove_overlapping_circles
from modules.utils import get_parameter
from modules.model.circle_detection.pre_processing import apply_erosion, apply_laplace, apply_opening

import os
import cv2 as cv
import math
import skimage.feature as skf

from sklearn.preprocessing import StandardScaler
import json

scaler = StandardScaler()

# Load picture and detect edges
def detect_circles(image_path):
    """
    Detects circles in an image using the Hough Transform algorithm.
    
    Parameters:
        image_path (str): The path to the image file.

    Returns:
        set: A set of tuples (center, radius) where center is also a tuple (x, y).
    """
    return detect_cicles_opencv(image_path)

def denormalize_1d(length, image_shape):
    return length * math.sqrt(image_shape[0] * image_shape[1])

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3
def CannyThreshold(val, src, src_gray):
    low_threshold = val
    img_blur = cv.blur(src_gray, (3,3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    return dst

def detect_cicles_opencv(image_path):
    print(image_path)
    src = cv.imread(image_path, cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        return -1
    
    ## Rescale the image to fit in a max_res x max_res window or keep the original size if it's smaller
    scale = 1
    max_largest_dim = get_parameter("hough_parameters")["max_res"]
    if src.shape[0] > max_largest_dim or src.shape[1] > max_largest_dim:
        scale = max_largest_dim / max(src.shape[0], src.shape[1])
        src = cv.resize(src, (int(src.shape[1] * scale), int(src.shape[0] * scale)))
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    gray = cv.medianBlur(gray, 5)
    # gray = cv.GaussianBlur(gray, (3, 3), 0)

    # Equalize image's histogram
    # gray = cv.equalizeHist(gray)

    parameters = get_hough_parameters()

    ### Pre-processing of the image
    if(parameters["pre_processing"]["apply_erosion"]):
        gray = apply_erosion(gray)
    if(parameters.get("apply_laplace")):
        gray = apply_laplace(gray)
    if(parameters["pre_processing"]["apply_opening"]):
        iterations = parameters["pre_processing"]["opening_iterations"]
        gray = apply_opening(gray, iterations)

    # Stack the images. src, mask, gray
    after_canny = CannyThreshold(0, src, gray)

    compare = np.hstack((src, cv.cvtColor(gray, cv.COLOR_GRAY2RGB), after_canny))
    
    rows = gray.shape[0]

    if(get_parameter("hough_parameters")["use_default_hough"]):
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8)
    else:
        min_radius = int(denormalize_1d(parameters["min_radius"], gray.shape))
        max_radius = int(denormalize_1d(parameters["max_radius"], gray.shape))
        minDist = int(denormalize_1d(parameters["minDist"], gray.shape))

        print(f"Running hough transform with parameters: {parameters}")

        print(f"Min radius: {min_radius}, Max radius: {max_radius}, minDist: {minDist}")
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT_ALT, parameters["dp"], minDist=minDist,
            param1=parameters["param1"], param2=parameters["param2"],
            minRadius=min_radius, maxRadius=max_radius)

    output = set()
    
    detected_coins_count = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = circles[0, :]

        if(get_parameter("hough_parameters")["post_processing"]["remove_overlapping"]):
            circles = remove_overlapping_circles(circles)

        for i in circles:
            print(f"Found circle! {i}")
            detected_coins_count += 1
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            output.add(((int(i[0] * 1 / scale), int(i[1] * 1 / scale)), int(radius * 1 / scale)))
            cv.circle(src, center, radius, (255, 0, 255), 3)

        cv.putText(src, f"Coins Detected: {detected_coins_count}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    if(get_parameter("hough_parameters")["show_preview"]):
        src = np.hstack((compare, src))
        cv.imshow(f"detected circles {image_path}", src)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return output

def get_hough_parameters():
    parameters = get_parameter("hough_parameters")
    
    if(parameters.get("param1") is None):
        parameters["param1"] = 100
    if parameters.get("param2") is None:
        parameters["param2"] = 30
    if parameters.get("min_radius") is None:
        parameters["min_radius"] = 1
    if parameters.get("max_radius") is None:
        parameters["max_radius"] = 30
    if parameters.get("dp") is None:
        parameters["dp"] = 1
    if parameters.get("minDist") is None:
        parameters["minDist"] = 30

    return parameters

def extract_color_and_hog_features(image_path, circles):
    src = cv.imread(image_path, cv.IMREAD_COLOR)
    features_list = []
    for (x, y, r, diameter) in circles:
        x1 = max(x - r, 0)
        y1 = max(y - r, 0)
        x2 = min(x + r, src.shape[1])
        y2 = min(y + r, src.shape[0])
        crop = src[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue 

        resized_crop = cv.resize(crop, (300, 300))

        lab_crop = cv.cvtColor(resized_crop, cv.COLOR_BGR2LAB)
        l_mean = np.mean(lab_crop[:, :, 0])
        a_mean = np.mean(lab_crop[:, :, 1])
        b_mean = np.mean(lab_crop[:, :, 2])
        
        hog_features, hog_image = extract_hog_features(resized_crop)  
        hog_features_1d = hog_features.reshape(-1)
        hog_features_normalized = StandardScaler().fit_transform(hog_features_1d.reshape(-1, 1))
        
        mean = np.mean(hog_features_normalized)
        variance = np.var(hog_features_normalized)

        print(f"Moyenne : {mean}")
        print(f"Variance : {variance}")

        features_list.append([diameter, l_mean, a_mean, b_mean, hog_features_normalized.tolist()])

        """ 
        plt.figure(figsize=(10, 10))
        plt.imshow(hog_image, cmap='gray')
        plt.title(f'HOG features for Circle at ({x}, {y})')
        plt.axis('off')
        plt.show() 
        """

    return features_list 
    
def extract_color_features(image_path, circles):
    src = cv.imread(image_path, cv.IMREAD_COLOR)
    features_list = []
    for (x, y, r, diameter) in circles:
        x1 = max(x - r, 0)
        y1 = max(y - r, 0)
        x2 = min(x + r, src.shape[1])
        y2 = min(y + r, src.shape[0])
        crop = src[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue  
        
        # normalization
        normalized_crop = crop.astype(np.float32) / 255.0
        resized_crop = cv.resize(normalized_crop, (10, 10))

        lab_crop = cv.cvtColor(resized_crop, cv.COLOR_BGR2LAB)
        l_mean = np.mean(lab_crop[:, :, 0])
        a_mean = np.mean(lab_crop[:, :, 1])
        b_mean = np.mean(lab_crop[:, :, 2])
        
        features_list.append([diameter, l_mean, a_mean, b_mean])

    return features_list

def extract_hog_features(region, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    gray_region = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
    hog_features, hog_image = skf.hog(gray_region, orientations=nbins, pixels_per_cell=cell_size, cells_per_block=block_size, visualize=True)
    return hog_features, hog_image


def create_features_vector(image_path):
    circles = detect_cicles_opencv(image_path)
    if circles.size > 0:
        color_features = extract_color_and_hog_features(image_path, circles)
        #color_features = extract_color_features(image_path, circles)
        return color_features
    else:
        return []
    
def save_to_json(data, file_name):
    def default_converter(o):
        if isinstance(o, np.integer):
            return int(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
    
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4, default=default_converter)
        
       
def load_images_and_labels(base_path):
    categories = os.listdir(base_path)
    data = []

    for category in categories:
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                img = cv.imread(str(img_path), cv.IMREAD_COLOR)
                if img is not None:
                    img_features = create_features_vector(str(img_path))
                    for feature in img_features:
                        data.append({
                            'features': feature,
                            'label': category
                        })

    return data