__authors__ = ['1632368', '1632367', '1632823']
__group__ = '172'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist
from utils import rgb2gray # Not sure if we can import


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)

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
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        ###Section 1: Reshape to (#images, 80p*60p)
        #print(test_data.shape) #(10, 80, 60)
        #Reshape test data (same as train data)
        if len(test_data.shape) == 3:
            # Already in Grayscale
            if test_data.shape[-1] == 3: # Not sure if we can import
                # If RGB, convert to grayscale # Not sure if we can import
                test_data = rgb2gray(test_data) # Not sure if we can import
            self.test_data = np.array(test_data.reshape((test_data.shape[0], -1)), dtype="float")
        #print(self.test_data.shape)#expected: (10, 4800) V
        
        ###Section 2: Distances
        distances = cdist(self.test_data, self.train_data, metric="euclidean")
        #print(distances)#not sure if they are correct
        
        ###Section 3: first neighbours
        #print(self.labels[1])
        neighbors = []
        for dist in distances:
            k_min_index = np.argsort(dist)[:k]
            #print(k_min_index)
            neighbors.append(self.labels[k_min_index])
            #print(self.labels[k_min_index])
        
        self.neighbors = np.array(neighbors)

    def get_class(self):
        """
        Get the class by maximum voting
        Args:
            None
        Return:
            1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #m = max elements
        #c = neighbour class
        
        neighbour_class = []    
        for neighbors in self.neighbors:
            #buscamos las clases sin repeticiones, los indices de cada clase y la cantidad de veces que se repite cada clase
            clas, index, inverse, counts = np.unique(neighbors, return_index=True, return_inverse=True, return_counts=True)
            #posiciones donde count es maximo
            m = np.where(counts == np.max(counts))[0]
            #cojemos el neighbor con el indice mas pequeño de entre los que count es maximo
            c = neighbors[index[m].min()]
            #añadimos a la lista
            neighbour_class.append(c)
        return np.array(neighbour_class)

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
