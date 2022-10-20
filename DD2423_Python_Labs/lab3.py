import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *
from gaussfft import gaussfft


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

    np.random.seed(seed)

    image_vec = image.reshape(-1, 3) * 1.

    centers = np.random.randint(image.min(), image.max(), (K, 3)) * 1.
    segmentation = np.zeros(image_vec.shape[0])

    for l in range(L):

        distances = distance_matrix(image_vec, centers)

        segmentation = np.argmin(distances, -1)
        for c in range(centers.shape[0]):
            if image_vec[segmentation == c].shape[0] > 0:
                centers[c] = np.mean( image_vec[segmentation == c], 0)

    return segmentation.reshape((image.shape[0], image.shape[1])), centers


def kmeans_segm_mix(image_vec, K, L, seed = 42):

    np.random.seed(seed)

    centers = np.random.randint(image_vec.min(), image_vec.max(), (K, 3)) * 1.
    segmentation = np.zeros(image_vec.shape[0])

    for l in range(L):

        real_centers = []

        distances = distance_matrix(image_vec, centers)

        segmentation = np.argmin(distances, -1)
        for c in range(centers.shape[0]):
            if image_vec[segmentation == c].shape[0] > 0:
                centers[c] = np.mean( image_vec[segmentation == c], 0)
                real_centers.append(c)

    return segmentation.astype(np.float32), centers[np.array(real_centers)].astype(np.float32), real_centers


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
    # image_masked_vec = (image[mask == 1]).reshape(-1, 3).astype(np.float32)
    # segmentation, mu_k, real_centers = kmeans_segm_mix(image_masked_vec, K, 2, seed=42)
    # K = len(real_centers)
    # for i, c in enumerate(real_centers):
    #     segmentation[segmentation == c] = i
    # sigma_k = np.repeat(np.expand_dims((np.eye(3) * 100.), 0), K, axis=0).astype(np.float32)
    # p_k = np.zeros((K, image_masked_vec.shape[0])).astype(np.float32)
    # g_k = np.zeros((K, image_masked_vec.shape[0])).astype(np.float32)
    # w_k = np.zeros(K).astype(np.float32)
    # for c in range(mu_k.shape[0]):
    #     w_k[c] = np.count_nonzero(segmentation == c) / np.sum(mask)
    #
    # for l in range(L):
    #     # g = (1. / np.sqrt((2*np.pi)**3 * np.linalg.det(sigma_k))) * np.exp()
    #     diff = distance_matrix(image_masked_vec, mu_k)
    #     for k in range(K):
    #         diff = image_masked_vec - mu_k[k]
    #         # diff = diff[segmentation == k]
    #         delta = np.diag(np.dot(np.dot(diff, np.linalg.inv(sigma_k[k])), diff.T))
    #         g_k[k] = (1. / np.sqrt((2*np.pi)**3 * np.linalg.det(sigma_k[k]))) * np.exp(-0.5 * delta)
    #         p_k[k] = w_k[k] * g_k[k]
    #     p_tot = np.sum(p_k, 0)
    #     p_k /= p_tot
    #
    #     w_k = np.mean(p_k, -1)
    #     for k in range(K):
    #         mu_k[k] = np.sum(image_masked_vec * np.expand_dims(p_k[k], -1), 0) / (np.sum(p_k[k]) + 1e-6)
    #         sigma_k[k] = np.dot(p_k[k] * diff.T, diff) / (np.sum(p_k[k]) + 1e-6)
    #
    # prob_vec = np.sum(np.expand_dims(w_k, -1) * g_k, 0)
    # prob = np.zeros((image.shape[0], image.shape[1]))
    # idx_x, idx_y = np.where(mask == 1)
    # for i, (x, y) in enumerate(zip(idx_x, idx_y)):
    #     prob[x, y] = prob_vec[i]

    image_masked_vec = (image[mask == 1]).reshape(-1, 3).astype(np.float32)

    # centers = np.random.randint(image.min(), image.max(), (K, 3)) * 1.

    segmentation, mu_k, real_centers = kmeans_segm_mix(image_masked_vec, K, 2, seed=42)
    K = len(real_centers)
    for i, c in enumerate(real_centers):
        segmentation[segmentation == c] = i

    sigma_k = np.repeat(np.expand_dims((np.eye(3) * 100.), 0), K, axis=0).astype(np.float32)
    p_k = np.zeros((K, image_masked_vec.shape[0])).astype(np.float32)
    g_k = np.zeros((K, image_masked_vec.shape[0])).astype(np.float32)

    w_k = np.zeros(K).astype(np.float32)
    for c in range(mu_k.shape[0]):
        w_k[c] = np.count_nonzero(segmentation == c) / np.sum(mask)

    for l in range(L):
        # g = (1. / np.sqrt((2*np.pi)**3 * np.linalg.det(sigma_k))) * np.exp()
        diff = distance_matrix(image_masked_vec, mu_k)
        for k in range(K):
            diff = image_masked_vec - mu_k[k]
            # diff = diff[segmentation == k]
            # delta = np.diag(np.dot(np.dot(diff, np.linalg.inv(sigma_k[k])), diff.T))
            delta = np.sum(np.dot(diff, np.linalg.inv(sigma_k[k])) * diff, -1)
            g_k[k] = (1. / np.sqrt((2 * np.pi) ** 3 * np.linalg.det(sigma_k[k]))) * np.exp(-0.5 * delta)
            p_k[k] = w_k[k] * g_k[k]
        p_tot = np.sum(p_k, 0)
        p_k /= p_tot

        w_k = np.mean(p_k, -1)
        for k in range(K):
            mu_k[k] = np.sum(image_masked_vec * np.expand_dims(p_k[k], -1), 0) / (np.sum(p_k[k]) + 1e-6)
            sigma_k[k] = np.dot(p_k[k] * diff.T, diff) / (np.sum(p_k[k]) + 1e-6)

    # prob_vec = np.sum(np.expand_dims(w_k, -1) * g_k, 0)
    # prob = np.zeros((image.shape[0], image.shape[1]))
    # idx_x, idx_y = np.where(mask == 1)
    # for i, (x, y) in enumerate(zip(idx_x, idx_y)):
    #     prob[x, y] = prob_vec[i]

    image_vec = image.reshape(-1, 3).astype(np.float32)
    g_o = np.zeros((K, image_vec.shape[0])).astype(np.float32)
    p_o = np.zeros((K, image_vec.shape[0])).astype(np.float32)
    for k in range(K):
        diff = image_vec - mu_k[k]
        delta = np.sum(np.dot(diff, np.linalg.inv(sigma_k[k])) * diff, -1)
        g_o[k] = (1. / np.sqrt((2 * np.pi) ** 3 * np.linalg.det(sigma_k[k]))) * np.exp(-0.5 * delta)
        p_o[k] = w_k[k] * g_o[k]
    prob = np.sum(p_o, 0).reshape(image.shape[0], image.shape[1])

    return prob