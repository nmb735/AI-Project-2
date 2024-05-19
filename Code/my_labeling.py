__authors__ = ['1632368', '1632367', '1632823']
__group__ = '172'

from utils_data import *
import numpy as np
from Kmeans import *
import Kmeans as Km
from KNN import *
import KNN as knn


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    def Retrieval_by_color(imgs, kmeans_color_labels, searched_color_s):
        found_imgs = []
        for image, color_label in zip(imgs, kmeans_color_labels):
            found_vector = np.isin(searched_color_s, color_label)
            found_bool = np.isin([True], found_vector)
            if found_bool[0]:
                found_imgs.append(image)
        found_imgs = np.array(found_imgs)
        return found_imgs
    
    def Retrieval_by_shape(imgs, knn_shape_labels, searched_shape_s):
        found_imgs = []
        for index, image in enumerate(imgs):
            #print(searched_shape_s)
            found_vector = np.isin(searched_shape_s, knn_shape_labels[index]) 
            #print(found_vector)
            found_bool = np.isin([True], found_vector)
            #print(found_bool)
            if found_bool:
                found_imgs.append(image)
        found_imgs = np.array(found_imgs)
        return found_imgs
    
    def Retrieval_combined(imgs, kmeans_color_labels, knn_shape_labels, searched_color_s, searched_shape_s):
        found_imgs = []
        new_shape_labels = []
        for image, color_label, shape_label in zip(imgs, kmeans_color_labels, knn_shape_labels):
            found_vector = np.isin(searched_color_s, color_label)
            found_bool = np.isin([True], found_vector)
            if found_bool[0]:
                found_imgs.append(image)
                new_shape_labels.append(shape_label)
        mid_search = np.array(found_imgs)
        new_shape_labels = np.array(new_shape_labels)
        #print(new_shape_labels)
        #visualize_retrieval(mid_search, mid_search.shape[0])
        final_search = Retrieval_by_shape(mid_search, new_shape_labels, searched_shape_s)
        return final_search
    
    '''
    # Define two arrays
    array1 = np.array([1, 2, 3, 4, 5])
    array2 = np.array([5, 6, 7, 8, 9])

    # Check if any value in array1 exists in array2
    result = np.isin(array1, array2)
    xd = np.isin([True], result)
    print(xd[0])
    
    if xd[0]:
        print('xd')
    '''
    
    #    SET Kmeans 
    colors_labels_list = []

    for img in cropped_images:
        km = KMeans(img)
        km.fit()
        colors_labels_list.append(get_colors(km.centroids))
    colors_labels_list = np.array(colors_labels_list)
    #print(colors_labels_list)
    #print(cropped_images.shape)
    #print(colors_labels_list.shape)
    
    #    SET KNN 
    #print(train_imgs.shape)
    #knn = KNN(train_imgs, train_class_labels)
    #shape_labels_list = knn.predict(imgs, 3)
    #shape_labels_list = np.array(shape_labels_list)
    train_imgs = train_imgs.reshape(train_imgs.shape[0], train_imgs.shape[1], train_imgs.shape[2] * train_imgs.shape[3])
    imgsknn = imgs.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2] * imgs.shape[3])
    knn = KNN(train_imgs, train_class_labels)
    shape_labels_list = knn.predict(imgsknn, 3)
    #print(shape_labels_list)
    #print(cropped_images.shape)
    #print(shape_labels_list.shape)
    
    
    
    
    
    # TEST: Retrieval_by_color
    
    searched_color_s = ['Blue', 'Red'] #modify this array for testing
    searched_color_s = np.array(searched_color_s)
    found_imgs = Retrieval_by_color(cropped_images, colors_labels_list, searched_color_s)
    #print(found_imgs.shape[0])
    #count = np.sum(colors_labels_list == 'Blue')
    #print(count)
    visualize_retrieval(found_imgs, found_imgs.shape[0])
    
    
    # TEST: Retrieval_by_shape
    '''
    searched_shape_s = ['Sandals'] #modify this array for testing
    searched_shape_s = np.array(searched_shape_s)
    found_imgs = Retrieval_by_shape(cropped_images, shape_labels_list, searched_shape_s)
    #print(found_imgs.shape[0])
    #count = np.sum(shape_labels_list == 'Sandals')
    #print(count)
    visualize_retrieval(found_imgs, found_imgs.shape[0])
    '''
    
    # TEST: Retrieval_combined
    '''
    searched_color_s = ['Black'] #modify this array for testing
    searched_color_s = np.array(searched_color_s)
    searched_shape_s = ['Shorts'] #modify this array for testing
    shape_labels_list = np.array(shape_labels_list)
    found_imgs = Retrieval_combined(imgs, colors_labels_list, shape_labels_list, searched_color_s, searched_shape_s)
    visualize_retrieval(found_imgs, found_imgs.shape[0])
    '''
    
    
    
    
    
    
