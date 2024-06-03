import cv2 as cv

def compute_f1(masked_ground_truth, masked_model):
    """
    Compute the F1 score given the ground truth and the model's output.
    """
    true_positives = cv.bitwise_and(masked_ground_truth, masked_model)
    false_positives = cv.subtract(masked_model, masked_ground_truth)
    false_negatives = cv.subtract(masked_ground_truth, masked_model)
    true_negatives = cv.bitwise_not(cv.bitwise_or(masked_ground_truth, masked_model))

    # Convert each to gray scale
    true_positives = cv.cvtColor(true_positives, cv.COLOR_BGR2GRAY)
    false_positives = cv.cvtColor(false_positives, cv.COLOR_BGR2GRAY)
    false_negatives = cv.cvtColor(false_negatives, cv.COLOR_BGR2GRAY)
    true_negatives = cv.cvtColor(true_negatives, cv.COLOR_BGR2GRAY)

    # Now count the number of pixels which are still 1
    true_positives_count = cv.countNonZero(true_positives)
    false_positives_count = cv.countNonZero(false_positives)
    false_negatives_count = cv.countNonZero(false_negatives)
    # true_negatives_count = cv.countNonZero(true_negatives)

    precision = true_positives_count / (true_positives_count + false_positives_count) if (true_positives_count + false_positives_count) > 0 else 0
    recall = true_positives_count / (true_positives_count + false_negatives_count) if (true_positives_count + false_negatives_count) > 0 else 0

    if(precision == 0 or recall == 0):
        return 0

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1

def compute_jaccard_index(masked_ground_truth, masked_model):
    """
    Compute the Jaccard index given the ground truth and the model's output.
    """
    intersection = cv.bitwise_and(masked_ground_truth, masked_model)
    union = cv.bitwise_or(masked_ground_truth, masked_model)
    intersection = cv.cvtColor(intersection, cv.COLOR_BGR2GRAY)
    union = cv.cvtColor(union, cv.COLOR_BGR2GRAY)

    intersection_count = cv.countNonZero(intersection)
    union_count = cv.countNonZero(union)

    jaccard_index = intersection_count / union_count

    return jaccard_index