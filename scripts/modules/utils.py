import json
import math

parameters = None

def reset_parameters():
    global parameters
    parameters = None

def get_parameters():
    global parameters  # Add global keyword to access the global variable
    if(parameters is None):
        with open("./parameters.json", "r") as file:
            parameters = json.loads(file.read())
    
    return parameters

def get_parameter(parameter_name):
        return get_parameters()[parameter_name]
    
def cut_image_in_square(image, center, size):
    """
    Given an image, a center and a size, return the square of the image centered at the center and with the given size.
    
    Args:
        image (np.array): The image to cut.
        center (tuple): The center of the square.
        size (int): The size of the square.
    
    Returns:
        np.array: The square of the image.
    """
    x, y = center
    x = int(x)
    y = int(y)
    
    # y = image.shape[0] - y  # Invert y axis

    half_size = int(size // 2)

    ## Calculate the new image slices taking into consideration out of bounds for the numpy array,
    ## with max() and min() functions

    i_start = max(0, y - half_size)
    i_end = min(image.shape[0], y + half_size)
    j_start = max(0, x - half_size)
    j_end = min(image.shape[1], x + half_size)

    return image[i_start:i_end, j_start:j_end]

def squared_distance(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def euclidean_distance(p1, p2):
    return squared_distance(p1, p2)**0.5