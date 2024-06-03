from os import listdir
from os.path import isfile, join
import matplotlib.image as mpimg
from random import shuffle
import math
import json
import sys
import os

sys.path.append("..")  # Add parent folder to sys.path

import modules.utils as utils


# Given all of the files in the image_pat,
# return the (training_set, validation_set, testing_set)
def split_datasets(image_path):
    propotions = (60, 20, 20) # Training, validation, testing, in percentage

    images = [f for f in listdir(image_path)
              if (isfile(join(image_path, f)) and ("DS_Store" not in f))]
    shuffle(images)

    propotions = (math.floor(propotions[0] / 100 * len(images)), math.floor(propotions[1] / 100 * len(images)), 0)
    propotions = (propotions[0], propotions[1], len(images) - propotions[0] - propotions[1])

    training_set = images[0:(propotions[0])]
    validation_set = images[propotions[0]:(propotions[0] + propotions[1])]
    testing_set = images[(propotions[0] + propotions[1]):]

    return (training_set, validation_set, testing_set)

def split_and_persist_datasets(data_sets):
    training_set, validation_set, testing_set = split_datasets(data_sets)

    with open("./parameters.json", "r") as file:
        parameters = json.loads(file.read())
        # Apend all three sets to the file
        parameters["training_set"] = training_set
        parameters["validation_set"] = validation_set
        parameters["testing_set"] = testing_set

    with open("./parameters.json", "w") as file:
        json.dump(parameters, file, indent = 4)

def get_labels():
    annotations_path = utils.get_parameter("annotations_path")

    labels = set()

    for f in listdir(annotations_path):
        file_path = join(annotations_path, f)
        if not isfile(join(annotations_path, f)) or not f.endswith(".json"):
            continue

        with open(file_path, "r") as file:
            shapes = json.loads(file.read())["shapes"]
            for shape in shapes:
                labels.add(shape["label"])

    return labels

def generate_coin_squares():
    image_path = utils.get_parameter("image_path")
    annotations_path = utils.get_parameter("annotations_path")
    for f in listdir(annotations_path):
        file_path = join(annotations_path, f)
        if not isfile(join(annotations_path, f)) or not f.endswith(".json"):
            continue

        training_set = [f.split(".")[0] for f in utils.get_parameter("training_set")]
        validation_set = [f.split(".")[0] for f in utils.get_parameter("validation_set")]
        testing_set = [f.split(".")[0] for f in utils.get_parameter("testing_set")]
        dataset = None

        file_number = f.split(".")[0]

        if file_number in training_set:
            dataset = "training_set"
        elif file_number in validation_set:
            dataset = "validation_set"
        elif file_number in testing_set:
            dataset = "testing_set"
        else:
            dataset = "other"

        with open(file_path, "r") as file:
            annotation = json.loads(file.read())
            shapes = annotation["shapes"]
            image_file_path = annotation["imagePath"]
            if "/" in image_file_path:
                image_file_path = image_file_path.split("/")[-1]
            
            if "\\" in image_file_path:
                image_file_path = image_file_path.split("\\")[-1]

            original_file_name = image_file_path.split(".")[:-1]
            original_file_name = "_".join(original_file_name)
            file_extension = image_file_path.split(".")[-1]

            full_image_path = join(image_path, image_file_path)
            print(full_image_path)
            image = mpimg.imread(full_image_path)
            
            i = 0
            error_count = 0

            for shape in shapes:
                center = tuple(shape["points"][0])
                radius = utils.euclidean_distance(center, shape["points"][1])
                label = shape["label"]

                square = utils.cut_image_in_square(image, center, 2 * radius)


                folder = f"./cache/{dataset}/{label}"

                # Create cache folder if it does not exist
                try:
                    os.makedirs(folder)
                except FileExistsError:
                    pass
                
                try:
                    mpimg.imsave(join(folder, f"{original_file_name}_{i}_{label}.{file_extension}"), square)
                except Exception as e:
                    print(f"Error: {e}")
                    error_count += 1

                i += 1


cache_coins_folder = "./cache/coins_detection"

def persist_coins_in_cache(image_name, coins):
    """
    Coins is a list of tuples as follows:
    - tuple: A tuple of tuples with the following structure:
        - label (str): The label of the coin.
        - center (tuple): The center coordinates of the coin (x, y).
        - radius (int): The radius of the coin.

    Save the information in the tuples in the cache folder as a csv
    """

    try:
        os.makedirs(cache_coins_folder)
    except FileExistsError:
        pass

    with open(f"{cache_coins_folder}/{image_name}.csv", "w") as file:
        file.write("center_x,center_y,radius\n")
        for coin in coins:
            file.write(f"{coin[0][0]},{coin[0][1]},{coin[1]}\n")

def cache_contains_coins(image_name):
    return isfile(f"{cache_coins_folder}/{image_name}.csv")

def load_coins_from_cache(image_name):

    if(not cache_contains_coins(image_name)):
        raise Exception(f"Cache does not contain coins for image {image_name}")

    coins = []
    with open(f"{cache_coins_folder}/{image_name}.csv", "r") as file:
        lines = file.readlines()
        for line in lines[1:]:
            parts = line.split(",")
            coins.append(((int(parts[0]), int(parts[1])), int(parts[2])))

    return coins