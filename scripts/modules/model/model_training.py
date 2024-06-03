from modules.utils import get_parameter, get_parameters, reset_parameters
from modules.model.model_tester import test_find_coins
import math
import json
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def squared_distance(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def euclidean_distance(p1, p2):
    return squared_distance(p1, p2)**0.5

def find_hough_parameters():
    training_set = get_parameter("training_set")

    parameters = get_parameters()

    found_params = find_param1_param2(training_set, parameters)

    parameters["hough_parameters"]["param1"] = float(found_params[0])
    parameters["hough_parameters"]["param2"] = float(found_params[1])

    with open("parameters.json", "w") as file:
        file.write(json.dumps(parameters, indent=4))
    
    return {
        "param_1": found_params[0],
        "param_2": found_params[1]
    }

def find_param1_param2(training_set, parameters):
    param_1_range = parameters["training"]["hough_parameters"]["param1"]
    param_2_range = parameters["training"]["hough_parameters"]["param2"]

    best_f1 = 0
    best_param_1 = 0
    best_param_2 = 0

    for i in np.arange(param_1_range["base"], param_1_range["bound"], param_1_range["step"]):
        for j in np.arange(param_2_range["base"], param_2_range["bound"], param_2_range["step"]):
            parameters["hough_parameters"]["param1"] = float(i)
            parameters["hough_parameters"]["param2"] = float(j)

            print(f"Testing parameters: param1 = {i}, param2 = {j}")

            with open("parameters.json", "w") as file:
                file.write(json.dumps(parameters, indent=4))

            reset_parameters()

            # We maximize the f1 score from the macro-average results
            f1_result = test_find_coins("training_set")["macro_average"]["f1"]

            if(f1_result > best_f1):
                best_f1 = f1_result
                best_param_1 = i
                best_param_2 = j

    return best_param_1, best_param_2
    

def find_min_max_radius(training_set):
    # Both of these are scaled by the image size
    min_radius = float('inf')
    max_radius = 0

    radius_array = []
    
    for image in training_set:
        print(f"Testing image: {image}")
        image_full_path = f"{get_parameter('image_path')}/{image}"
        annotated_image_path = f"{get_parameter('annotations_path')}/{image.split('.')[0]}.json"

        with open(annotated_image_path, "r") as file:
            shapes = json.loads(file.read())['shapes']

        for shape in shapes:
            center = tuple(map(int, shape['points'][0]))  # Convert center to integers
            radius = int(euclidean_distance(center, shape['points'][1]))

            img_shape = mpimg.imread(image_full_path).shape
            normalized_radius = normalize_radius(radius, img_shape)

            radius_array.append(normalized_radius)

            min_radius = min(min_radius, normalized_radius)
            max_radius = max(max_radius, normalized_radius)

    min_radius = np.percentile(radius_array, 10)
    print(f"Min radius: {min_radius}, Max radius: {max_radius}")

    return {
        "min_radius": min_radius,
        "max_radius": max_radius
    }

def normalize_radius(radius, image_shape):
    return radius / math.sqrt(image_shape[0] * image_shape[1])

def denormalize_radius(radius, image_shape):
    return radius * math.sqrt(image_shape[0] * image_shape[1])

