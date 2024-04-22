import pickle
from KNN import KNN

def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':
    label_map = { 0: 'bottom-left', 1 : 'bottom-right', 2: 'top-left', 3: 'top-right'}

    data = load_data_from_pickle('testCases_img_knn/knn_wooclap_dataset.pkl')
    knn = KNN(data['train_images'], data['train_labels'])

    # Test1: Check the total number of images in the training data
    knn_train_data = knn.train_data
    print("Images number on the:", knn_train_data.shape[0])

    # Test2: Check the total pixel size of images in the training data.
    knn_train_data = knn.train_data
    print("Images total pixels size:", knn_train_data.shape[1])

    # Test3: Check the total number of test images evaluated to determine neighbor labels
    knn.get_k_neighbours(data['test_images'],2)
    knn_neighbours_labels = knn.neighbors
    print("Amount of testing images evaluated:", knn_neighbours_labels.shape[0])

    # Test4: Check the number of neighbor labels selected for each test image using K=2
    knn.get_k_neighbours(data['test_images'], 2)
    knn_neighbours_labels = knn.neighbors
    print("Closest labels selected amount:", knn_neighbours_labels.shape[1])

    # Test5: Check the predicted label for the first test image using K=4
    label = knn.predict(data['test_images'], 4)
    print("Label for k=4: ", label_map[label[0]])

    # Test6: Check the predicted label for the last test image using K=6
    label = knn.predict(data['test_images'], 2)
    print("Label for k=6: ", label_map[label[3]])