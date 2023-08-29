import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, utils
from sklearn.metrics import confusion_matrix, recall_score, f1_score, roc_auc_score, precision_score, accuracy_score

def load_dataset(dataset_folder, class_folders):
    images = []
    labels = []
    for i, class_folder in enumerate(class_folders):
        for filename in os.listdir(os.path.join(dataset_folder, class_folder)):
            img = cv2.imread(os.path.join(dataset_folder, class_folder, filename))
            img = cv2.resize(img, (150, 150))  # resize images
            img = img / 255.0  # normalize pixel values
            images.append(img)
            labels.append(i)
    return np.array(images), utils.to_categorical(np.array(labels))

dataset_folder = "C:\\Users\\Dogu\\Downloads\\dataset_lesion"
class_folders = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

images, labels = load_dataset(dataset_folder, class_folders)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_folders), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Performance metrics calculation
conf_matrix = confusion_matrix(true_classes, predicted_classes)
recall = recall_score(true_classes, predicted_classes, average='macro')
f1 = f1_score(true_classes, predicted_classes, average='macro')
roc_auc = roc_auc_score(test_labels, predictions, multi_class='ovr')
precision = precision_score(true_classes, predicted_classes, average='macro')
accuracy = accuracy_score(true_classes, predicted_classes)

# Output performance metrics
print('Confusion Matrix:')
print(conf_matrix)
print('\nRecall Score:', recall)
print('F1 Score:', f1)
print('ROC AUC Score:', roc_auc)
print('Precision Score:', precision)
print('Classification Accuracy:', accuracy)

# Creating and writing the results to excel file
results_df = pd.DataFrame({'True Class': true_classes, 'Predicted Class': predicted_classes})
results_df.to_excel("results.xlsx", engine='openpyxl')

metrics_df = pd.DataFrame({'Recall Score': [recall], 'F1 Score': [f1], 'ROC AUC Score': [roc_auc], 'Precision Score': [precision], 'Classification Accuracy': [accuracy]})
metrics_df.to_excel("metrics.xlsx", engine='openpyxl')
