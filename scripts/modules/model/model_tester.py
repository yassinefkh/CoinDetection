import json
import numpy as np
import sys
import cv2 as cv
import matplotlib.pyplot as plt

sys.path.append("..")  # Add parent folder to sys.path

from modules.utils import get_parameter, get_parameters
from modules.model.metrics_util import compute_jaccard_index
from modules.model.model_wrapper import detect_coins
from modules.model.model_wrapper import find_coins

label_mapping = {
    "50cts": "50_centimes",
    "20cts": "20_centimes",
    "10cts": "10_centimes",
    "5cts": "5_centimes",
    "2cts": "2_centimes",
    "1cts": "1_centime",
    "1e": "1_euro",
    "2e": "2_euro",
    "2_euros": "2_euro",
    "1_centimes": "1_centime",
    "1_centime_inverse": "1_centime",
    "50_centimes_inverse": "50_centimes",
    "missed": "missed"
}

short_label_mapping = {
    "50_centimes": "50cts",
    "20_centimes": "20cts",
    "10_centimes": "10cts",
    "5_centimes": "5cts",
    "2_centimes": "2cts",
    "1_centime": "1cts",
    "1_euro": "1e",
    "2_euro": "2e",
    "2_euros": "2e",
    "1_centimes": "1cts",
    "1_centime_inverse": "1cts",
    "50_centimes_inverse": "50cts",
    "missed": "missed"
}

short_to_value_mapping = {
    "missed": 0,
    "not_detected": 0,
    "50cts": 0.5,
    "20cts": 0.2,
    "10cts": 0.1,
    "5cts": 0.05,
    "2cts": 0.02,
    "1cts": 0.01,
    "1e": 1,
    "2e": 2,
}

def get_coin_value(short_label):
    return short_to_value_mapping.get(short_label, 0)

def shorten_label(label):
    """Convert label using predefined mapping."""
    return short_label_mapping.get(label, label)

def clean_label(label):
    """Convert label using predefined mapping."""
    return label_mapping.get(label, label)

def squared_distance(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def euclidean_distance(p1, p2):
    return squared_distance(p1, p2)**0.5

def detected_coins_contains(detected_coins, label, center):
    """
    Check if any of the detected coins contains a specific label and center,
    i.e. the labels should match and the expected center should be within the radius of the detected coin.

    Args:
        detected_coins (list): A list of dictionaries representing the detected coins from the model.
        label (str): The label to search for.
        center (tuple): The center coordinates to compare with.

    Returns:
        tuple: A tuple of the form (label, center, radius) if the coin is found, else None.
    """
    for coin in detected_coins:
        if (coin['label'] == label and
            (squared_distance(coin['center'], center) < coin['radius']**2)):
            return coin
    return None

def find_circle_given_truth(ground_truth, circle, original_image_shape):
    """
    Given a set of ground truth circles and a detected circle,
    return the ground truth circle that corresponds to the detected circle
    or None if no corresponding ground truth circle is found.

    Returns:
        truth: Corresponding ground truth circle if the detected circle is found, else None.
    """

    for truth in ground_truth:    
        if(is_same_circle(circle, truth, original_image_shape)):
            return truth

    return None

def is_same_circle(circle, truth, original_image_shape, threshold=0.5):
    """
    Return True if the jaccard index between the detected circle and at least one of the ground truth circles
    is greater than the threshold, else False.

    Args:
        ground_truth (set): A set of tuples (center, radius).
        circle (tuple): A tuple (center, radius).
    """
    circle_mask = np.zeros(original_image_shape, np.uint8)
    cv.circle(circle_mask, circle[0], circle[1], (255, 255, 255), -1)

    truth_mask = np.zeros(original_image_shape, np.uint8)
    cv.circle(truth_mask, truth[0], truth[1], (255, 255, 255), -1)

    jaccard_index = compute_jaccard_index(truth_mask, circle_mask)
    return jaccard_index > threshold

def is_the_same_detected_coin(detected_coin, circle, original_image_shape, threshold=0.5):
    """
    Given a detected coin and a ground truth circle, return True if the detected coin is the same as the ground truth circle,
    i.e. the jaccard index between the detected coin and the ground truth circle is greater than the threshold, else False.

    Args:
        detected_coin (dict): A tuple (label, center, radius).
        circle (dict): A tuple (label, center, radius).

    Returns:
        bool: True if the detected coin is the same as the ground truth circle, else False.
    """
    circle_mask = np.zeros(original_image_shape, np.uint8)
    cv.circle(circle_mask, circle[1], circle[2], (255, 255, 255), -1)

    detected_mask = np.zeros(original_image_shape, np.uint8)
    cv.circle(detected_mask, detected_coin[1], detected_coin[2], (255, 255, 255), -1)

    jaccard_index = compute_jaccard_index(circle_mask, detected_mask)

    return jaccard_index > threshold

# Test the find_coins() method by computing the intersection with what we expect from the annotated images
# 1. Load the testing dataset from the annotated JSON files.
def test_find_coins(dataset = "validation_set", parameters = get_parameters()):
    testing_data = get_parameter(dataset)

    micro_average_tp = 0
    micro_average_fp = 0
    micro_average_fn = 0

    macro_average_precision_sum = 0
    macro_average_recall_sum = 0

    for image in testing_data:
        print(f"\nTesting image: {image}")
        image_full_path = f"{get_parameter('image_path')}/{image}"
        annotated_image_path = f"{get_parameter('annotations_path')}/{image.split('.')[0]}.json"

        detected_coins = find_coins(image_full_path, parameters)
        true_positives_count = 0
        false_positives_count = 0
        false_negatives_count = 0
        # true_negatives_count = 0

        ground_truth = set()

        with open(annotated_image_path, "r") as file:
            shapes = json.loads(file.read())['shapes']

            image_data = cv.imread(image_full_path)

            ## Create blank image with the same size as the original
            masked_ground_truth = np.zeros((image_data.shape[0], image_data.shape[1], 3), np.uint8)

            for shape in shapes:
                center = tuple(map(int, shape['points'][0]))  # Convert center to integers
                label = shape['label']
                radius = int(euclidean_distance(center, shape['points'][1]))

                cv.circle(masked_ground_truth, center, radius, (255, 255, 255), -1)
                ground_truth.add((center, radius))

                side_by_side = np.hstack((image_data, masked_ground_truth))

        masked_model = np.zeros((image_data.shape[0], image_data.shape[1], 3), np.uint8)

        for coin in detected_coins:
            center = coin[0]
            radius = coin[1]
            cv.circle(masked_model, center, radius, (255, 255, 255), -1)

            found_truth = find_circle_given_truth(ground_truth, coin, image_data.shape)

            if(found_truth is not None):
                true_positives_count+=1
                ground_truth.remove(found_truth)
            else:
                false_positives_count+=1

        false_negatives_count = len(ground_truth)

        # Add to the global metrics
        micro_average_tp += true_positives_count
        micro_average_fp += false_positives_count
        micro_average_fn += false_negatives_count

        precision = true_positives_count / (true_positives_count + false_positives_count) if (true_positives_count + false_positives_count) > 0 else 0
        recall = true_positives_count / (true_positives_count + false_negatives_count) if (true_positives_count + false_negatives_count) > 0 else 0
        f1_score = 0 if (precision == 0 or recall == 0) else 2 * (precision * recall) / (precision + recall)

        macro_average_precision_sum += precision
        macro_average_recall_sum += recall

        print(f"TP: {true_positives_count}, FP: {false_positives_count}, FN: {false_negatives_count}")
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1_score}")
        side_by_side = np.hstack((image_data, masked_model))
        # cv.imshow("side_by_side", side_by_side)
        # cv.waitKey(0)

    # Compute global precision, recall and F1 score with the micro average method
    micro_average_precision = micro_average_tp / (micro_average_tp + micro_average_fp) if (micro_average_tp + micro_average_fp) > 0 else 0
    micro_average_recall = micro_average_tp / (micro_average_tp + micro_average_fn)
    
    if(micro_average_precision == 0 or micro_average_recall == 0):
        micro_average_f1 = 0
    else:
        micro_average_f1 = 2 * (micro_average_precision * micro_average_recall) / (micro_average_precision + micro_average_recall)

    print(f"Micro-Average results:")
    print(f"TP: {micro_average_tp}, FP: {micro_average_fp}, FN: {micro_average_fn}")
    print(f"Precision: {micro_average_precision}, Recall: {micro_average_recall}, F1: {micro_average_f1}")

    # Compute global precision, recall and F1 score with the macro average method
    macro_average_precision = macro_average_precision_sum / len(testing_data)
    macro_average_recall = macro_average_recall_sum / len(testing_data)

    if(macro_average_precision == 0 or macro_average_recall == 0):
        macro_average_f1 = 0
    else:
        macro_average_f1 = 2 * (macro_average_precision * macro_average_recall) / (macro_average_precision + macro_average_recall)

    print(f"Macro-Average results:")
    print(f"Precision: {macro_average_precision}, Recall: {macro_average_recall}, F1: {macro_average_f1}")
    
    return {
        "micro_average": {
            "precision": micro_average_precision,
            "recall": micro_average_recall,
            "f1": micro_average_f1
        },
        "macro_average": {
            "precision": macro_average_precision,
            "recall": macro_average_recall,
            "f1": macro_average_f1
        }
    }

def find_test_coin_in_predicted_set(predicted_coins, test_coin, original_image_shape, threshold=0.5):
    """
    Given a set of predicted coins and a test coin, return the predicted coin that matches the test coin.

    Args:
        predicted_coins (list): A list of tuples (center, radius).
        test_coin (tuple): A tuple (center, radius).
        original_image_shape (tuple): The shape of the original image.
        threshold (float): The threshold for the Jaccard index.
    """
    for found_coin in predicted_coins:
        found_coin_label = found_coin[0]
        test_coin_label = test_coin[0]
        if(is_the_same_detected_coin(found_coin, test_coin, original_image_shape, threshold)
                and found_coin_label == test_coin_label):
            return found_coin

    return None

def find_coin_with_same_circle(predicted_coins, coin, original_image_shape, threshold=0.5):
    coin_circle = (coin[1], coin[2])

    for found_coin in predicted_coins:
        found_coin_circle = (found_coin[1], found_coin[2])

        if is_same_circle(found_coin_circle, coin_circle, original_image_shape, threshold):
            return found_coin

    return None

def get_rescale_factor(src, max_largest_dim):
    scale = 1
    max_largest_dim = get_parameter("hough_parameters")["max_res"]
    if src.shape[0] > max_largest_dim or src.shape[1] > max_largest_dim:
        scale = max_largest_dim / max(src.shape[0], src.shape[1])
        src = cv.resize(src, (int(src.shape[1] * scale), int(src.shape[0] * scale)))

    return (src, scale)

def rescaled_coordinate(coordinate, rescale_factor):
    return (int(coordinate[0] * rescale_factor), int(coordinate[1] * rescale_factor))

def rescaled_dimension(dimension, rescale_factor):
    return int(dimension * rescale_factor)

# type can be correct_prediction, wrong_classification, or missed_coin
def draw_text_with_background(img, text, center, radius, type):
    fontFace = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.75
    thickness = 1

    baseline = 0
    textSize, _ = cv.getTextSize(text, fontFace, fontScale, thickness)
    baseline += thickness

    if type == "correct_prediction":
        textOrg = (center[0] - radius, center[1] - radius - textSize[1])
        background_color = (0, 255, 0)  # Green
        text_color = (0, 0, 0)  # Black
    elif type == "wrong_classification":
        textOrg = (center[0] + 2 * radius - textSize[0], center[1] + radius + textSize[1] + 12)
        background_color = (0, 0, 255)  # Red
        text_color = (255, 255, 255)  # White
    elif type == "missed_coin":
        textOrg = (center[0] - radius, center[1] - radius - textSize[1])
        background_color = (255, 0, 0)  # Blue
        text_color = (255, 255, 255)  # White
    else:
        textOrg = (center[0] - radius, center[1] - radius - textSize[1])
        background_color = (255, 255, 255)  # White
        text_color = (0, 0, 0)  # Black

    cv.rectangle(img, (textOrg[0], textOrg[1] + baseline), (textOrg[0] + textSize[0], textOrg[1] - textSize[1]), background_color, -1)
    cv.line(img, (textOrg[0], textOrg[1] + thickness), (textOrg[0] + textSize[0], textOrg[1] + thickness), background_color)

    thickness = int(thickness)  # Ensure thickness is an integer

    cv.putText(img, text, textOrg, fontFace, fontScale, text_color, thickness, cv.LINE_8)

def annotate_image_with_results_and_display(img, correct_coins, incorrecly_labeled_coins, missed_coins):
    """
    Annotate the image with the detected coins and the ground truth coins and display the image.
    Also include the label by the coin

    Args:
        img (numpy.ndarray): The input image.
        detected_coins (list): A list of detected coins.
        shapes (list): A list of shapes from the ground truth.
    """

    if not get_parameters()["coin_recognition"]["display_evaluation_after_recognition"]:
        return
    
    # Reshape image to fit in a 1024x1024 box
    img, rescale_factor = get_rescale_factor(img, 1024)

    for coin in correct_coins:
        center = rescaled_coordinate(coin[1], rescale_factor)
        radius = rescaled_dimension(coin[2], rescale_factor)
        cv.circle(img, center, radius, (0, 255, 0), 2)
        draw_text_with_background(img, coin[0], center, radius, "correct_prediction")

    for coin in incorrecly_labeled_coins:
        center = rescaled_coordinate(coin[1], rescale_factor)
        radius = rescaled_dimension(coin[2], rescale_factor)
        cv.circle(img, center, radius, (0, 0, 255), 2)
        draw_text_with_background(img, coin[0], center, radius, "wrong_classification")

    if get_parameters()["coin_recognition"]["evaluation_display_ground_truth"]:
        for coin in missed_coins:
            center = rescaled_coordinate(coin[1], rescale_factor)
            radius = rescaled_dimension(coin[2], rescale_factor)
            cv.circle(img, center, radius, (255, 0, 0), 2)
            draw_text_with_background(img, coin[0], center, radius, "missed_coin")

    cv.imshow("Annotated image", img)
    cv.waitKey(0)

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot the confusion matrix where the y axis is the true labels and the x axis is the predicted labels.
    The true labels should include a "not_a_coin" value that matches to a certain value in the predicted labels
    when a coin was detected with the coin detection algorithm but that wasn't a coin in the ground truth.
    """
    all_labels = y_true + y_pred

    # Map all to their equivalent short value
    all_labels = ["unmatch" if label == "missed" or label == "not_a_coin" else label for label in all_labels]
    all_labels = [shorten_label(label) for label in all_labels]
    all_labels = list(set(all_labels))
    y_true = [shorten_label(label) for label in y_true]
    y_pred = [shorten_label(label) for label in y_pred]
    
    # Sort using get_coin_value
    all_labels.sort(key=get_coin_value)

    # For indexing purposes, any missed or not_a_coin is "unmatch" in all_labels
    y_true_labels = [label if label != "unmatch" else "not_a_coin" for label in all_labels]
    y_pred_labels = [label if label != "unmatch" else "missed" for label in all_labels]
    y_pred_labels.sort(key=get_coin_value)
    y_true_labels.sort(key=get_coin_value)

    confusion_matrix = np.zeros((len(all_labels), len(all_labels)))

    for i in range(len(y_true)):
        true_label = y_true[i] if y_true[i] != "not_a_coin" else "unmatch"
        pred_label = y_pred[i] if y_pred[i] != "missed" else "unmatch"

        true_label_index = all_labels.index(true_label)
        pred_label_index = all_labels.index(pred_label)

        confusion_matrix[true_label_index][pred_label_index] += 1

    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)

    ax.set_xticks(np.arange(len(all_labels)))
    ax.set_yticks(np.arange(len(all_labels)))

    ax.set_xticklabels(y_pred_labels)
    ax.set_yticklabels(y_true_labels)

    # Add the title for each axis
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(all_labels)):
        for j in range(len(all_labels)):
            text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="w")

    ax.set_title("Confusion matrix")
    fig.tight_layout()
    plt.show()

    # Sum diagonal and divide by total number of elements
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

def shapes_to_coins(shapes):
    coins = []
    for shape in shapes:
        center = tuple(shape['points'][0])
        center = tuple(map(int, center))
        label = clean_label(shape['label'])
        radius = int(euclidean_distance(center, shape['points'][1]))
        coins.append((label, center, radius))
    return coins

def compute_class_wise_results(correct_predictions_count_map, class_instances_count_map, class_predictions):
    labels_sorted_by_value = sorted(correct_predictions_count_map.keys(), key=lambda x: get_coin_value(shorten_label(x)))

    result = {}

    for label in labels_sorted_by_value:
        precision = correct_predictions_count_map[label] / class_predictions[label]
        recall = correct_predictions_count_map[label] / class_instances_count_map[label]
        f1_score = 0 if (precision == 0 or recall == 0) else 2 * (precision * recall) / (precision + recall)

        print(f"Label: {label}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {f1_score}")
        print("\n")

        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    
    return result

def compute_macro_average(results):
    precision_sum = 0
    recall_sum = 0
    f1_score_sum = 0

    for label in results:
        precision_sum += results[label]["precision"]
        recall_sum += results[label]["recall"]
        f1_score_sum += results[label]["f1_score"]

    precision = precision_sum / len(results)
    recall = recall_sum / len(results)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Macro-average precision: {precision}")
    print(f"Macro-average recall: {recall}")
    print(f"Macro-average F1 score: {f1_score}")

    return (precision, recall, f1_score)

def plot_results(results):
    """
    Plot the results of the model testing using a heatmap

    Args:
        results (dict): A dictionary of the results, indexed by label and containing the precision, recall, and F1 score.
    """
    labels = list(results.keys())
    precision = [results[label]["precision"] for label in labels]
    recall = [results[label]["recall"] for label in labels]
    f1_score = [results[label]["f1_score"] for label in labels]
    labels = list(map(shorten_label, labels))

    x = np.arange(len(labels))
    fig, ax = plt.subplots()

    im = ax.imshow([precision, recall, f1_score], interpolation='nearest')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')  # Rotate x labels by 45 degrees

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Precision', 'Recall', 'F1 Score'])

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Scores', rotation=-90, va="bottom")

    plt.show()

def test_model(dataset="validation_set"):
    testing_dataset = get_parameter(dataset)

    all_labels = set()
    y_true = []
    y_pred = []

    correctly_classified = 0
    total = 0

    correct_predictions_count_map = {}
    class_instances_count_map = {} # Denominator for recall
    class_predictions_count_map = {} # Denominator for precision

    for image in testing_dataset:
        print(f"\nTesting image: {image}")
        detected_coins = detect_coins(image, get_parameters())
        print(f"Detected coins: {detected_coins}")

        annotated_image_path = f"{get_parameter('annotations_path')}/{image.split('.')[0]}.json"
        image_full_path = f"{get_parameter('image_path')}/{image}"

        img_data = cv.imread(image_full_path)

        for coin in detected_coins: ## Add all class predictions to denominator count
            class_predictions_count_map[coin[0]] = class_predictions_count_map.get(coin[0], 0) + 1

        with open(annotated_image_path, "r") as file:
            shapes = json.loads(file.read())['shapes']
            truth_coins = shapes_to_coins(shapes)
            found_coins = set()  # A set of tuples (label, center, radius)
            not_found_coins = set()  # Idem

            for coin in truth_coins:
                label = coin[0]
                all_labels.add(label)
                image_shape = img_data.shape

                correctly_found_and_labeled_coin = find_test_coin_in_predicted_set(detected_coins, coin, image_shape)

                # Add all class instances to the recall denominator count
                class_instances_count_map[label] = class_instances_count_map.get(label, 0) + 1

                if correctly_found_and_labeled_coin is not None: # Meaning coin was found and labeled correctly
                    found_coins.add(correctly_found_and_labeled_coin)
                    correct_predictions_count_map[label] = correct_predictions_count_map.get(label, 0) + 1
                    detected_coins.remove(correctly_found_and_labeled_coin)  # Each coin should only be matched once
                    y_true.append(label)
                    y_pred.append(correctly_found_and_labeled_coin[0])

                else: # Coin was not found or was not labeled correctly
                    not_found = coin
                    not_found_coins.add(not_found)

            remaining_found_coins = detected_coins
            for coin in remaining_found_coins: ## For each coin found but not correctly labeled
                all_labels.add(coin[0])
                equivalent_coin = find_coin_with_same_circle(not_found_coins, coin, img_data.shape)

                if equivalent_coin is not None: # There is a real coin with the same circle
                    y_true.append(equivalent_coin[0])
                    y_pred.append(coin[0])
                    not_found_coins.remove(equivalent_coin)
                else: # There is no coin with the same circle
                    y_true.append("not_a_coin")
                    y_pred.append(coin[0])

            remaining_truth_coins = not_found_coins
            for coin in remaining_truth_coins:
                y_true.append(coin[0])
                y_pred.append("missed")

            print(f"Found coins: {found_coins}")
            print(f"Missed coins: {not_found_coins}")

            correctly_classified += len(found_coins)
            total += len(found_coins) + len(not_found_coins) + len(detected_coins)
            
            annotate_image_with_results_and_display(img_data, found_coins, detected_coins, not_found_coins)

    print("\n\nClass-wise results:")
    results = compute_class_wise_results(correct_predictions_count_map, class_instances_count_map, class_predictions_count_map)

    print("\n\nOverall results:")
    print(f"Correctly classified: {correctly_classified}")
    print(f"Total: {total}")
    print(f"Accuracy: {correctly_classified / total}")

    plot_confusion_matrix(y_true, y_pred)
    plot_results(results)

    precision, recall, f1 = compute_macro_average(results)

    return {
        "accuracy": correctly_classified / total,
    }

