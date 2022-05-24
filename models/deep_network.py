from keras.models import Sequential
from keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import config as cfg


class DeepNetwork:

    def __init__(self, train_data: ImageDataGenerator, val_data: ImageDataGenerator,
                 n_classes: int, model_type: str):
        self.train_data = train_data
        self.val_data = val_data
        self.n_classes = n_classes
        self.model_type = model_type
        model = self._select_model()
        self.model, self.history = self.train_model(model)

    def _select_model(self):
        model = []
        if self.model_type == "cnn":
            model = self.cnn_model()
        elif self.model_type == "inception":
            model = self.inception_v3()
        return model

    def cnn_model(self):
        # create model
        model = Sequential()
        model.add(Conv2D(20, (3, 3), input_shape=(cfg.row, cfg.cox, 3), activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1024, activation='sigmoid'))
        model.add(Dense(self.n_classes, activation='sigmoid'))
        # Compile model
        model.compile(loss=cfg.loss, optimizer=cfg.optimizer, metrics=cfg.metrics)
        return model

    def inception_v3(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)
        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, activation='softmax'))
        model.add(Dense(self.n_classes, activation='sigmoid'))
        model.layers[0].trainable = False
        return model

    def train_model(self, model):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

        history = model.fit(self.train_data, epochs=30, validation_data=self.val_data,
                            callbacks=[es])

        return model, history
