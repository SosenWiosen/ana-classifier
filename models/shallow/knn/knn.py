import os
import cv2  # OpenCV for image manipulation
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler


from model_tester.graph.export_confusion_matrix import export_confusion_matrix
from model_tester.save_report import save_classification_report


def load_images_from_folder(folder):
    images = []
    labels = []
    class_names = []
    label = 0
    for subfolder in sorted(os.listdir(folder)):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            class_names.append(subfolder)  # Capture class name from the folder name.
            for filename in sorted(os.listdir(subfolder_path)):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    red_channel = img[:, :, 2]
                    red_channel = cv2.resize(red_channel, (28, 28))
                    images.append(red_channel.flatten())
                    labels.append(label)
            label += 1
    return np.array(images), np.array(labels), class_names

folder_path = '/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/NON-STED/train'
test_path = '/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/NON-STED/test'
X, y, class_names = load_images_from_folder(folder_path)
X = X / 255.0  # Normalize pixel values to 0-1

# Apply Random Over Sampling to balance the classes
ros = RandomOverSampler(random_state=123)
X_resampled, y_resampled = ros.fit_resample(X, y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=123)

test_x, test_y, _ = load_images_from_folder(test_path)
test_x = test_x / 255.0  # Normalize pixel values to 0-1

# Initialize the KNN classifier
neighbor_settings = range(1, 50)  # Example: Test from 1 to 10 neighbors
accuracies = []
test_accuracies = []

# Ensure output directories exist
if not os.path.exists('results'):
    os.makedirs('results')

# Loop over different values of k
for n_neighbors in neighbor_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_val)
    test_predictions = knn.predict(test_x)
    accuracy = accuracy_score(y_val, predictions)
    test_accuracy = accuracy_score(test_y, test_predictions)
    accuracies.append(accuracy)
    test_accuracies.append(test_accuracy)

    print(f"Accuracy for k={n_neighbors}: {accuracy}")
    print(f"Test Accuracy for k={n_neighbors}: {test_accuracy}")

    # Save the classification report
    report_file_path = f"results_28/classification_report_{n_neighbors}.json"
    save_classification_report(y_val, predictions, report_file_path, class_names)
    save_classification_report(test_y, test_predictions, f"results_28/test_classification_report_{n_neighbors}.json", class_names)

    # Save the confusion matrix
    export_confusion_matrix(y_val, predictions, class_names, "results_28", f"cfm_{n_neighbors}")
    export_confusion_matrix(test_y, test_predictions, class_names, "results_28", f"test_cfm_{n_neighbors}")
