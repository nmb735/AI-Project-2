__authors__ = ['1632368', '1632367', '1632823']
__group__ = ''

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist
from utils import rgb2gray # Not sure if we can import


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        Initializes the train data
        Args:
            train_data: PxMxNx3 matrix corresponding to P color images
        Return: 
            assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        if len(train_data.shape) == 3:
            # Already in Grayscale
            if train_data.shape[-1] == 3: # Not sure if we can import
                # If RGB, convert to grayscale # Not sure if we can import
                train_data = rgb2gray(train_data) # Not sure if we can import
            self.train_data = np.array(train_data.reshape((train_data.shape[0], -1)), dtype="float")
            
    def get_k_neighbours(self, test_data, k):
        """
        Given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        Args:
            test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
            k: the number of neighbors to look at
        Return:
            the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.neighbors = np.random.randint(k, size=[test_data.shape[0], k])

    def get_class(self):
        """
        Get the class by maximum voting
        Args:
            None
        Return:
            1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return np.random.randint(10, size=self.neighbors.size), np.random.random(self.neighbors.size)

    def predict(self, test_data, k):
        """
        Predicts the class at which each element in test_data belongs to
        Args:
            test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
            k: the number of neighbors to look at
        Return:
            the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
