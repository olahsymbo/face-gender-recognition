import os
import inspect

app_path = inspect.getfile(inspect.currentframe())
directory = os.path.realpath(os.path.dirname(app_path))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn import metrics

from utils.preprocess import DataLoader
from models.deep_network import DeepNetwork
from utils.confusion_matrix import plot_confusion_matrix
import config as cfg

# create training dataset with label

train_dateset, train_label = DataLoader.dataloader(cfg.TRAINING_FOLDER + cfg.male_folder, cfg.label_male)
training_dataset_female, training_label_female = DataLoader.dataloader(cfg.TRAINING_FOLDER + cfg.female_folder,
                                                                       cfg.label_female)

train_dateset.extend(training_dataset_female)
train_label.extend(training_label_female)

# Splitting the feature and label for train dataset
empty_arr = [i for i, x in enumerate(train_dateset) if x == []]
feature_train = [i for i in train_dateset if i != []]
feature_train = np.array(feature_train).reshape(-1, cfg.row, cfg.cox, 3)
labels_train = [i for i in train_label]
labels_train = np.array(labels_train).astype(str)

val_dateset, val_label = DataLoader.dataloader(cfg.VAL_FOLDER + cfg.male_folder, cfg.label_male)
val_dataset_female, val_label_female = DataLoader.dataloader(cfg.VAL_FOLDER + cfg.female_folder, cfg.label_female)

val_dateset.extend(val_dataset_female)
val_label.extend(val_label_female)

# Splitting the feature and label for val dataset
empty_arr = [i for i, x in enumerate(val_dateset) if x == []]
feature_val = [i for i in val_dateset if i != []]
feature_val = np.array(feature_val).reshape(-1, cfg.row, cfg.cox, 3)
labels_val = [i for i in val_label]
labels_val = np.array(labels_val).astype(str)

testing_dataset, test_label = DataLoader.dataloader(cfg.TESTING_FOLDER + cfg.male_folder, cfg.label_male)
testing_dataset_female, testing_label_female = DataLoader.dataloader(cfg.TESTING_FOLDER + cfg.female_folder,
                                                                     cfg.label_female)

testing_dataset.extend(testing_dataset_female)
test_label.extend(testing_label_female)

# Splitting the feature and label for test dataset
# feature_test = [i for i in test_dataset]
empty_arr = [i for i, x in enumerate(testing_dataset) if x == []]
feature_test = [i for i in testing_dataset if i != []]
feature_test = np.array(feature_test).reshape(-1, cfg.row, cfg.cox, 3)
labels_test = [i for i in test_label]
labels_test = np.array(labels_test).astype(str)

# label encoding
number = preprocessing.LabelEncoder()

labels_train = number.fit_transform(labels_train)
labels_val = number.fit_transform(labels_val)
labels_test = number.fit_transform(labels_test)

y_train = pd.DataFrame(labels_train)
y_val = pd.DataFrame(labels_val)
y_test = pd.DataFrame(labels_test)

n_classes = y_train.shape[1]

# Data Augmentation

datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest',
                             rescale=1. / 255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = datagen.flow_from_directory(
    cfg.TRAINING_FOLDER,  # This is the source directory for training images
    target_size=(cfg.row, cfg.cox),  # All images will be resized to 150x150
    batch_size=60,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

# Flow validation images in batches of 20 using val_datagen generator
validation_generator = datagen.flow_from_directory(
    cfg.VAL_FOLDER,
    target_size=(cfg.row, cfg.cox),
    batch_size=60,
    class_mode='binary')

test_generator = datagen.flow_from_directory(
    cfg.TESTING_FOLDER,
    target_size=(cfg.row, cfg.cox),
    batch_size=60,
    class_mode='binary')

img = load_img('/content/dataset1/train/man/face_1493.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1):
    i += 1
    plt.figure()
    plt.imshow(batch.reshape(x.shape[1], x.shape[2], 3))
    if i > 20:
        break  # otherwise the generator would loop indefinitely

# Using transfer learning with Inception V3 pretrained model

y_train = pd.DataFrame(labels_train)
y_val = pd.DataFrame(labels_val)
y_test = pd.DataFrame(labels_test)

n_classes = y_train.shape[1]

# define the inception model and added globalaveragepool, 2 fully connected layer.
# the top layer is fixed not trainable to maintain the weights of inception net.
dpn = DeepNetwork(train_data=train_generator, val_data=validation_generator,
                  n_classes=n_classes, model_type="inception")
model, history = dpn.model

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()
scores = model.predict(feature_test)
print("CNN Accuracy: ", metrics.accuracy_score(y_test, scores.round()) * 100)

# Plot the Confusion matrix
cnf_matrix = confusion_matrix(y_test, scores.round())
np.set_printoptions(precision=2)

# create the names of classes for confusion matrix plot
class_names = np.array(['Male', 'Female'])

# Plot normalized confusion matrix
plt.figure()
plt.figure(figsize=(40, 8))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')