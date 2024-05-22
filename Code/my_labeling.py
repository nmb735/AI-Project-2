__authors__ = ['1632368', '1632367', '1632823']
__group__ = '172'

from utils_data import *
import numpy as np
from Kmeans import *
import Kmeans as Km
from KNN import *
import KNN as knn
import time

#########################################################################################
# Quality analysis functions
#########################################################################################
def Retrieval_by_color(imgs, kmeans_color_labels, searched_color_s):
    """
    Retrieves images from a list based on their color labels.

    Args:
        imgs (list): A list of images to search through.
        kmeans_color_labels (list): A list of K-means color labels corresponding to the images.
        searched_color_s (list): A list of color labels to search for.

    Returns:
        np.array: An array of images that contain any of the colors in `searched_color_s`.
    """
    found_imgs = []
    for image, color_label in zip(imgs, kmeans_color_labels):
        found_vector = np.isin(searched_color_s, color_label)
        found_bool = np.isin([True], found_vector)
        if found_bool[0]:
            found_imgs.append(image)
    found_imgs = np.array(found_imgs)
    return found_imgs

def Retrieval_by_shape(imgs, knn_shape_labels, searched_shape_s):
    """
    Retrieves images from a list based on their shape labels.

    Args:
        imgs (list): A list of images to search through.
        knn_shape_labels (list): A list of KNN shape labels corresponding to the images.
        searched_shape_s (list): A list of shape labels to search for.

    Returns:
        np.array: An array of images that contain any of the shapes in `searched_shape_s`.
    """

    found_imgs = []
    for index, image in enumerate(imgs):
        found_vector = np.isin(searched_shape_s, knn_shape_labels[index]) 
        found_bool = np.isin([True], found_vector)
        if found_bool:
            found_imgs.append(image)
    found_imgs = np.array(found_imgs)
    return found_imgs

def Retrieval_combined(imgs, kmeans_color_labels, knn_shape_labels, searched_color_s, searched_shape_s):
    """
    Retrieves images from a list based on their color and shape labels.

    Args:
        imgs (list): A list of images to search through.
        kmeans_color_labels (list): A list of K-means color labels corresponding to the images.
        knn_shape_labels (list): A list of KNN shape labels corresponding to the images.
        searched_color_s (list): A list of color labels to search for.
        searched_shape_s (list): A list of shape labels to search for.

    Returns:
        np.array: An array of images that contain any of the colors in `searched_color_s` and any of the shapes in `searched_shape_s`.
    """
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
    final_search = Retrieval_by_shape(mid_search, new_shape_labels, searched_shape_s)
    return final_search


############################################################################################################
# Quantitative analysis functions
############################################################################################################
def Kmean_statistics(img, k_max, options=None):
    """
    Computes and plots the statistics for K-Means clustering on a set of images.

    Args:
        img (list): A list of images to perform K-Means clustering on.
        k_max (int): The maximum number of clusters to consider.
        options (dict, optional): A dictionary of options to pass to the KMeans constructor. Defaults to None.

    Returns:
        tuple: A tuple containing three NumPy arrays. The first array contains the average time taken for each value of k, 
               the second contains the average within-class distance for each value of k, 
               and the third contains the average number of iterations for each value of k.
    """
    k_time_axis = np.zeros(k_max - 1)
    k_wcd_axis = np.zeros(k_max - 1)
    k_iterations_axis = np.zeros(k_max - 1)
    n_images = len(img)

    for i in img:
        for k in range(2, k_max+1):
            kmeans = KMeans(i, k, options)
            start_time = time.time()
            kmeans.fit()
            end_time = time.time()
            k_time_axis[k-2] += (end_time - start_time)
            k_wcd_axis[k-2] += kmeans.withinClassDistance()
            k_iterations_axis[k-2] += kmeans.num_iterations

    k_time_axis /= n_images
    k_wcd_axis /= n_images
    k_iterations_axis /= n_images

    # Plotting
    k_values = list(range(2, k_max+1))

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Average time taken', color=color)
    ax1.plot(k_values, k_time_axis, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Average within-class distance', color=color)
    ax2.plot(k_values, k_wcd_axis, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()
    color = 'tab:green'
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Average number of iterations', color=color)
    ax3.plot(k_values, k_iterations_axis, color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

    return k_time_axis, k_wcd_axis, k_iterations_axis
    
def Get_shape_accuracy(shape_results, ground_truth):
    """
    Calculates the accuracy of shape predictions.

    Args:
        shape_results (np.array): An array of predicted shape labels.
        ground_truth (np.array): An array of ground truth shape labels.

    Returns:
        float: The accuracy of the shape predictions, in percent.
    """
    return (np.sum(shape_results == ground_truth) / len(ground_truth)) * 100

def Get_color_accuracy(color_results, ground_truth):
    """
    Calculates the accuracy of color predictions.

    This function compares the predicted color labels (`color_results`) with the ground truth labels (`ground_truth`).
    It calculates the accuracy as the percentage of correct predictions, with partial credit given for partially correct predictions.
    The function also calculates the mean number of colors per image in the ground truth.

    Args:
        color_results (list): A list of predicted color labels. Each label is a list of colors.
        ground_truth (list): A list of ground truth color labels. Each label is a list of colors.

    Returns:
        tuple: A tuple containing two floats. The first is the accuracy of the color predictions, in percent. 
               The second is the mean number of colors per image in the ground truth.
    """
    correct = 0
    for color, gt in zip(color_results, ground_truth):
        clr = color[0]
        if clr:
            if len(gt) == 1:
                if clr == gt[0]:
                    correct += 1
            else:
                partial = 0
                size = len(gt)
                if size > 0:
                    for c in gt:
                        if clr == c:
                            partial += 1
                    correct += (partial / size)
        else:
            if not gt: 
                correct += 1
    mean = np.mean([len(gt) for gt in ground_truth])
    return ((correct / len(ground_truth)) * 100), mean

def Get_total_accuracy(shape_results, color_results, ground_truth_shape, ground_truth_color):
    """
    Calculates the total accuracy of shape and color predictions.

    This function calculates the total accuracy as the average of the shape accuracy and color accuracy.

    Args:
        shape_results (np.array): An array of predicted shape labels.
        color_results (list): A list of predicted color labels. Each label is a list of colors.
        ground_truth (tuple): A tuple containing the ground truth shape labels and color labels.

    Returns:
        float: The total accuracy of the predictions, in percent.
    """
    shape_accuracy = Get_shape_accuracy(shape_results, ground_truth_shape)
    color_accuracy = Get_color_accuracy(color_results, ground_truth_color)
    return (shape_accuracy + color_accuracy) / 2

#########################################################################################
# Testing Function
#########################################################################################
def testing(train_imgs): # Modify this function for testing - General Template
    time_list = []
    total_accuracy_list = []
    shape_accuracy_list = []
    color_accuracy_list = []
    start_time = time.time()
    options = {}
    # SET Kmeans 
    colors_labels_list = []
    for img in cropped_images:
        km = KMeans(img,1,options)
        km.find_bestK(10)
        km.fit()
        colors_labels_list.append(get_colors(km.centroids))
    colors_labels_list = np.array(colors_labels_list)
    # SET KNN 
    train_imgs = train_imgs.reshape(train_imgs.shape[0], train_imgs.shape[1], train_imgs.shape[2] * train_imgs.shape[3])
    imgsknn = imgs.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2] * imgs.shape[3])
    knn = KNN(train_imgs, train_class_labels)
    shape_labels_list = knn.predict(imgsknn, 3)

    end_time = time.time()
    time_list.append(end_time - start_time)
    total_accuracy_list.append(Get_total_accuracy(shape_labels_list, colors_labels_list, class_labels, color_labels))
    shape_accuracy_list.append(Get_shape_accuracy(shape_labels_list, class_labels))
    color_accuracy_list.append(Get_color_accuracy(colors_labels_list, color_labels))

    for i in zip(time_list, total_accuracy_list, shape_accuracy_list, color_accuracy_list):
        print(f"Execution {i+1}")
        print(f"Time: {i[0]}\nTotal Accuracy: {i[1]}\nShape Accuracy: {i[2]}\nColor Accuracy: {i[3]}\n")
    

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)
    
    # SET Kmeans 
    colors_labels_list = []

    for img in cropped_images:
        km = KMeans(img)
        km.fit()
        colors_labels_list.append(get_colors(km.centroids))
    colors_labels_list = np.array(colors_labels_list)
    
    # SET KNN 
    train_imgs = train_imgs.reshape(train_imgs.shape[0], train_imgs.shape[1], train_imgs.shape[2] * train_imgs.shape[3])
    imgsknn = imgs.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2] * imgs.shape[3])
    knn = KNN(train_imgs, train_class_labels)
    shape_labels_list = knn.predict(imgsknn, 3)

    # TEST & Example: Retrieval_by_color
    searched_color_s = ['Blue', 'Red'] #modify this array for testing
    searched_color_s = np.array(searched_color_s)
    found_imgs = Retrieval_by_color(cropped_images, colors_labels_list, searched_color_s)
    visualize_retrieval(found_imgs, found_imgs.shape[0])
    
    # TEST: Retrieval_by_shape
    searched_shape_s = ['Sandals'] #modify this array for testing
    searched_shape_s = np.array(searched_shape_s)
    found_imgs = Retrieval_by_shape(cropped_images, shape_labels_list, searched_shape_s)
    visualize_retrieval(found_imgs, found_imgs.shape[0])

    # TEST: Retrieval_combined
    searched_color_s = ['Black'] #modify this array for testing
    searched_color_s = np.array(searched_color_s)
    searched_shape_s = ['Shorts'] #modify this array for testing
    shape_labels_list = np.array(shape_labels_list)
    found_imgs = Retrieval_combined(imgs, colors_labels_list, shape_labels_list, searched_color_s, searched_shape_s)
    visualize_retrieval(found_imgs, found_imgs.shape[0])

    # TEST: Get_shape_accuracy
    print(f"SHAPE ACCURACY TEST\n {Get_shape_accuracy(shape_labels_list, class_labels)}%")
    
    # TEST: Get_color_accuracy
    x, m = Get_color_accuracy(colors_labels_list, color_labels)
    max_per = (1/m)*100
    print(f"COLOR ACCURACY TEST\n {x}%, with m = {m}\n Maximum percentatge: {max_per}%\n Relative Percentatge: {(x/max_per)*100}%")
    
    # TEST: Kmean_statistics
    x,y,z = Kmean_statistics(imgs,5)
    print(f"KMEAN STATISTICS\n Time: {x}\n WCD: {y}\n Iterations: {z}")