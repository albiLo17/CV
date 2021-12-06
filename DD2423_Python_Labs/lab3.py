import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *
from gaussfft import gaussfft
import os
from tqdm import tqdm

from PIL import Image

ex = 2

def kmeans_segm(image, K, L, seed = 42):
    """
    Implement a function that uses K-means to find cluster 'centers'
    and a 'segmentation' with an index per pixel indicating with 
    cluster it is associated to.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        seed - random seed
    Output:
        segmentation: an integer image with cluster indices
        centers: an array with K cluster mean colors
    """

    # Random initialization of the centers
    np.random.seed(seed)
    centers = np.random.randint(image.min(),image.max(),(K,3))
    segmentation = np.zeros((image.shape[0], image.shape[1]))

    # Compute all distances
    distances = np.zeros((image.shape[0], image.shape[1], K))
    for i in range(image.shape[0]):
        distances[i] = scipy.spatial.distance_matrix(image[i], centers)

    for iteration in tqdm(range(L)):
        # Compute clusters for each point
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                cluster = np.argmax(distances[x][y])
                segmentation[x][y] = int(cluster)

        # Update centers
        for k in range(K):
            if k in segmentation:
                indeces = np.where(segmentation == k)
                num_points = len(indeces[0])
                center_val = np.zeros((3))
                for j in range(num_points):
                    center_val += image[indeces[0][j]][indeces[1][j]]
                centers[k] = center_val/ num_points

        # Compute all distances
        distances = np.zeros((image.shape[0], image.shape[1], K))
        for i in range(image.shape[0]):
            distances[i] = scipy.spatial.distance_matrix(image[i], centers)

    return segmentation.astype(int), centers


def mixture_prob(image, K, L, mask):
    """
    Implement a function that creates a Gaussian mixture models using the pixels 
    in an image for which mask=1 and then returns an image with probabilities for
    every pixel in the original image.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        mask - an integer image where mask=1 indicates pixels used 
    Output:
        prob: an image with probabilities per pixel
    """ 
    return prob

# if ex==2:
#     img = Image.open('Images-jpg/orange.jpg')
#     I = np.asarray(img).astype(np.float32)
#
#     print()