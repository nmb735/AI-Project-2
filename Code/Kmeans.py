__authors__ = ['1632368', '1632367', '1632823']
__group__ = ''

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
        Constructor of KMeans class

        Args:
            X (list or np.array): list(matrix) of all pixel values
            K (int): Number of cluster. 1 by default.
            options (dict): dictionary with options. None by default.
        """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        X = X.astype(float)

        if X.ndim > 2:
            X = X.reshape(-1, 3)

        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed

    def _init_centroids(self):
        """
        Initialization of centroids for K-Means algorithm
        Args:
            None
        """
        pixel_rep_const = 255
        self.old_centroids = np.zeros((self.K, self.X.shape[1]))
        self.centroids = np.zeros((self.K, self.X.shape[1]))

        if self.options['km_init'] == 'first':
            unique_pixels = set()
            extracted_pixels = []
            for pixel in self.X:
                pixel_tuple = tuple(pixel)
                if pixel_tuple not in unique_pixels:
                    unique_pixels.add(pixel_tuple)
                    extracted_pixels.append(pixel)
                    if len(extracted_pixels) == self.K:
                        break
            self.centroids = np.array(extracted_pixels)
        elif self.options['km_init'] == 'random':
            unique_pixels = set()
            while len(unique_pixels) < self.K:
                pixel = tuple(np.random.randint(0, pixel_rep_const, size=(self.X.shape[1])))
                unique_pixels.add(pixel)
            self.centroids = np.array(list(unique_pixels))

        elif self.options['km_init'] == 'custom': # Custom initialization: random, duplicates allowed
            self.centroids = np.random.randint(0, pixel_rep_const, size=(self.K, self.X.shape[1]))

        else:
            pass

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid.
        Args:
            None
        """
        self.labels = np.argmin((distance(self.X, self.centroids)), axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        Args:
            None
        """
        self.old_centroids = np.copy(self.centroids)

        # Check if there is any cluster without points and calculate the new centroids
        if np.any(self.labels) > 0:
            for i in np.arange(self.K):
                self.centroids[i] = np.mean(self.X[self.labels == i], axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids.
        Args:
            None
        """
        return np.allclose(self.centroids, self.old_centroids, self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        Args:
            None
        """
        self.num_iter = 0
        self._init_centroids()

        while self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            if self.converges():
                break
            self.num_iter += 1
        

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        WCD = 0
        for i in range(self.K):
            #Agafem els elements propers al centroid
            son_propers = self.labels == i;
            #Calculem distancia entre l'element proper i el centroid
            distancies = self.X[son_propers]-self.centroids[i]
            #Elevem aquest valor al cuadrat
            distanciesE2 = np.power(distancies, 2)
            #Fem la suma per fila
            distancia = np.sum(distanciesE2, axis = 1)
            #Suma total
            WCD += np.sum(distancia)
            #print(WCD)
        #Dividim per el nombre d'elements
        WCD = WCD / len(self.X)
        self.WCD = WCD
        return WCD 

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        #Apliquem K-Means una primera vegada amb K = 2
        self.K = 2
        self.fit()
        #Calculem WCD de K = 2
        WCD_anterior = self.withinClassDistance()
        
        i = 2 #=K
        DEC = 0
        #Executem el bucle fins que trobem la K optima o ens pasem de K maxima
        while (i < max_K) and ((100-DEC) >= 20):
            i += 1
            #Apliquem K-Means amb K per comparar amb amb el K-Means amb K-1
            self.K = i
            self.fit()
            #Calculem WCD de K
            WCD = self.withinClassDistance()
            #Calculem DEC amb WCD de K i WWCD de K-1
            DEC = 100*(WCD/WCD_anterior)
            #print(WCD_anterior)
            #print(WCD)
            #print(DEC)
            #El nou valor de WCD ara es l'anterior
            WCD_anterior = WCD
        #Si hem trobat una K optima, hem d'executar K-Means amb K-1, ja que K es pasa del nostre criteri
        if (i != max_K):
            self.K = i-1
            self.fit()
            return i
        #Sino tornem la maxima
        else:
            return max_K

# Out-of-class functions
def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    return np.sqrt(np.sum((X[:, np.newaxis, :] - C) ** 2, axis=2))


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    #print(centroids)
    
    #calculem les probabilitats del centroide
    color_prob = utils.get_color_prob(centroids)
    #print(color_prob)
    #trobem la posició de l'element més gran de les probabilitats, amb axis=1 busquem l'element major de la fila
    pos_max = np.argmax(color_prob, axis = 1)
    #print(pos_max)
    #busquem el color que representa la probabilitat més alta
    labels = utils.colors[pos_max]
    #print(labels)
    
    return labels
