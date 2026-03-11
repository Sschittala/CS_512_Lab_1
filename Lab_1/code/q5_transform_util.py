
# utility files used to apply rotations and transformations to data, used in q5 

import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy

# load in training data, store in dicionary
# {
#     'id': 1,
#     'letter': 'a',
#     'next_id': 2,
#     'word_id': 1, WILL USE TO APPLY SPECIFIC TRANSFORMATION
#     'position': 1,
#     'pixels': np.array([0, 0, 1, 0, 0, 1, 0, ...], dtype=np.uint8)
# } 
def load_train_data(filename):
    letters = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            letter_info = {
                'id': int(parts[0]),
                'letter': parts[1],
                'next_id': int(parts[2]),
                'word_id': int(parts[3]),
                'position': int(parts[4]),
                'pixels': np.array([int(p) for p in parts[5:]], dtype=np.uint8)
            }
            letters.append(letter_info)
    return letters

# apply the transform to a provided letter
def apply_transform(pixel_vector, transform):

    # reshape row-major: 16 rows x 8 cols
    img = pixel_vector.reshape((16, 8))

    # rotation
    if transform[0] == 'r':
        angle = transform[1]
        center = (8/2, 16/2)  # (cols/2, rows/2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        transformed_img = cv2.warpAffine(img, M, (8,16), flags=cv2.INTER_LINEAR, borderValue=0)
    
    # "slide" transformation
    elif transform[0] == 't':
        dx, dy = transform[1], transform[2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        transformed_img = cv2.warpAffine(img, M, (8,16), flags=cv2.INTER_NEAREST, borderValue=0)

    # no change
    else:
        transformed_img = img

    return transformed_img.flatten()

# loop through all letters and apply transformations from transform_dict
def transform_train_letters(letters, transform_dict):
    for letter in letters:
        wid = letter['word_id']
        if wid in transform_dict:
            letter['pixels'] = apply_transform(letter['pixels'], transform_dict[wid])
    return letters

# save the transformed file
def save_transformed_train(letters, filename):
    with open(filename, 'w') as f:
        for l in letters:
            pixel_str = ' '.join(str(int(p)) for p in l['pixels'])
            line = f"{l['id']} {l['letter']} {l['next_id']} {l['word_id']} {l['position']} {pixel_str}\n"
            f.write(line)
