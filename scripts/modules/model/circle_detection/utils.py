import numpy as np
from modules.utils import euclidean_distance

def remove_overlapping_circles(circles):
    # Find circles whose center is below a certain THRESHOLD of another
    # and then choose the one with the highest radius (the outer circle)

    circles = sorted(circles, key=lambda x: x[2], reverse=True)
    to_delete = set()
    
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            # Since circles is sorted in descending order of radius,
            # circle i will always have a higher radius than circle j
            center_i = np.int32(circles[i][:2])
            center_j = np.int32(circles[j][:2])

            dist = euclidean_distance(center_i, center_j)

            if(dist < circles[i][2]): # Meaning that circle j is inside circle i
                to_delete.add(j)

    return np.delete(circles, list(to_delete), axis=0)

