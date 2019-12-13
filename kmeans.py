from matplotlib import pyplot as io
import numpy as np
import numpy.ma as ma
from PIL import Image
import copy

'''
    Load file, file -> ndarray:
'''

# Image path:
Koala = 'Koala.jpg'
Penguins = 'Penguins.jpg'

'''
    Functions needed:
'''

'''
    Initial centers according to K in K-means:
    Input: K in K-means
    Output: Center points
'''


def initial_centers(k):
    return np.random.randint(0, 255, size=(k, 3)).astype(np.uint8)


'''
    Classify all pixels according to initial center points.
    Input: initial center points, image points.
    Output: pixel class matrix corresponding to pixel matrix of images.
'''


def classify(init, img_data):
    # Generate class matrix egg:
    class_mat = np.zeros((np.shape(img_data)[0], np.shape(img_data)[1]))

    # Initial dis_mat to record mix distance of each pixel.
    # distance for each pixel can not exceeds 500 in theory. So dis_mat at begin has a max distance.
    dis_mat = np.ones((np.shape(img_data)[0], np.shape(img_data)[1]))
    dis_mat *= 250000

    #
    for i in range(len(init)):
        # Deep copy image data:
        data_mat = copy.deepcopy(img_data)

        # Calculate pixels' distance to each center, generate final distance matrix of pixels:

        # difference matrix:
        tmp_dis_mat = data_mat - init[i]
        # distance matrix (square distance):
        tmp_dis_mat = tmp_dis_mat * tmp_dis_mat
        tmp_dis_mat = np.sum(tmp_dis_mat, -1)
        # min distance matrix: new_dis_mat:
        new_dis_mat = np.minimum(tmp_dis_mat, dis_mat)
        # 0: distance not changed:
        mask = dis_mat - new_dis_mat
        # Assign class label to pixels:
        class_mat = ma.array(class_mat, mask=mask, dtype='int32', fill_value=(i + 1))
        class_mat = class_mat.filled()
        # Update dis_mat to new distances:
        dis_mat = new_dis_mat

    return class_mat


'''
    Calculate new center according to the classification:
    Input: class_mat, image_mat, K (must be consist to function 'initial_centers').
    Output: new center points.
'''


def new_centers(k, class_mat, image_mat):
    center = np.empty((0, 3))
    for c in range(k):
        mask = (class_mat == (c + 1))
        mask = ~mask
        mask = ma.dstack((mask, mask, mask))
        tmp_image = ma.array(image_mat, mask=mask, fill_value=0)
        tmp_image = tmp_image.filled()

        # Get average center coordinate:
        center = np.append(center, [np.mean(tmp_image, axis=(0, 1))], axis=0)

    return center


'''
    Main:
    K_means clustering:
'''

# Load data:
img_Koala = io.imread(Koala)
img_Penguins = io.imread(Penguins)

# Load parameters K:
Ks = np.array([2, 5, 10, 15, 20])

for K in Ks:
    # Arbitrary initial centers:
    centers = initial_centers(K)

    # Classification from arbitrary initial centers:
    new_classes = classify(centers, img_Koala)

    # Arbitrary initial classification:
    classes = np.ndarray(np.shape(new_classes))

    while ~((classes == new_classes).all()):
        classes = new_classes
        centers = new_centers(K, new_classes, img_Koala)
        new_classes = classify(centers, img_Koala)

    print('K = ', K)
    unique_elements, counts_elements = np.unique(classes, return_counts=True)

    # Assign RGB to each class:
    for i in unique_elements:
        mask_ = ma.array(classes == i)
        f = np.ones(np.shape(classes))
        mask_f = ma.array(f == 0)
        color = centers[i - 1]

        mask1 = np.dstack((mask_, mask_f, mask_f))
        color1 = color[0].astype(np.uint8)
        img_Koala = ma.array(img_Koala, mask=mask1, dtype=np.uint8, fill_value=color1)
        img_Koala = img_Koala.filled()

        mask2 = np.dstack((mask_f, mask_, mask_f))
        color2 = color[1].astype(np.uint8)
        img_Koala = ma.array(img_Koala, mask=mask2, dtype=np.uint8, fill_value=color2)
        img_Koala = img_Koala.filled()

        mask3 = np.dstack((mask_f, mask_f, mask_))
        color3 = color[2].astype(np.uint8)
        img_Koala = ma.array(img_Koala, mask=mask3, dtype=np.uint8, fill_value=color3)
        img_Koala = img_Koala.filled()
        # Print new images:
        img = Image.fromarray(img_Koala)
        path = 'C:/Users/User/Desktop/Computer Science/6375 machine learning/Assignments/Assignment_5/' + \
               'K Koala/Koala' + str(K) + '.png'
        img.save(path)


for K in Ks:
    # Arbitrary initial centers:
    centers = initial_centers(K)

    # Classification from arbitrary initial centers:
    new_classes = classify(centers, img_Penguins)

    # Arbitrary initial classification:
    classes = np.ndarray(np.shape(new_classes))

    while ~((classes == new_classes).all()):
        classes = new_classes
        centers = new_centers(K, new_classes, img_Penguins)
        new_classes = classify(centers, img_Penguins)

    print('K = ', K)
    unique_elements, counts_elements = np.unique(classes, return_counts=True)

    # Assign RGB to each class:
    for i in unique_elements:
        mask_ = ma.array(classes == i)
        f = np.ones(np.shape(classes))
        mask_f = ma.array(f == 0)
        color = centers[i - 1]

        mask1 = np.dstack((mask_, mask_f, mask_f))
        color1 = color[0].astype(np.uint8)
        img_Penguins = ma.array(img_Penguins, mask=mask1, dtype=np.uint8, fill_value=color1)
        img_Penguins = img_Penguins.filled()

        mask2 = np.dstack((mask_f, mask_, mask_f))
        color2 = color[1].astype(np.uint8)
        img_Penguins = ma.array(img_Penguins, mask=mask2, dtype=np.uint8, fill_value=color2)
        img_Penguins = img_Penguins.filled()

        mask3 = np.dstack((mask_f, mask_f, mask_))
        color3 = color[2].astype(np.uint8)
        img_Penguins = ma.array(img_Penguins, mask=mask3, dtype=np.uint8, fill_value=color3)
        img_Penguins = img_Penguins.filled()
        # Print new images:
        img = Image.fromarray(img_Penguins)
        path = 'C:/Users/User/Desktop/Computer Science/6375 machine learning/Assignments/Assignment_5/' + \
               'K Penguins/Penguin' + str(K) + '.png'
        img.save(path)

