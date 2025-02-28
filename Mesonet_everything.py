from tensorflow.keras import Model                                                                                                                                                                                                  # type: ignore
from tensorflow.keras import Input                                                                                                                                                                                    # type: ignore
from tensorflow.keras.layers import Conv2D, ReLU, ELU, LeakyReLU, Dropout, Dense, MaxPooling2D, Flatten, BatchNormalization, Input                                                                                                                                                                                    # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator                                                                                                                                                                                    # type: ignore
from tensorflow.keras.optimizers import Adam                                                                                                                                                                                    # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping                                                                                                                                                                                    # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay                                                                                                                                                                                    # type: ignore                                                                                                                                                                                  # type: ignore
from tensorflow.keras.models import load_model                                                                                                                                                                                    # type: ignore
from sklearn.metrics import classification_report                                                                                                                                                                                    # type: ignore                                                                                                                                                                                  # type: ignore                                                                                                                                                                                 # type: ignore
from tensorflow.keras.models import Model                                                                                                                                                                                    # type: ignore                                                                                                                                                                                # type: ignore
import numpy as np                                                                                                                                                                                    # type: ignore
import matplotlib.pyplot as plt                                                                                                                                                                                    # type: ignore
from mpl_toolkits.axes_grid1 import ImageGrid                                                                                                                                                                                     # type: ignore
from math import floor, log
from datetime import datetime
import os
import pickle

IMG_WIDTH = 256

def get_datagen(use_default_augmentation=True, **kwargs):
    kwargs.update({'rescale': 1./255})
    if use_default_augmentation:
        kwargs.update({
            'rotation_range': 15,
            'zoom_range': 0.2,
            'brightness_range': (0.8, 1.2),
            'channel_shift_range': 30,
            'horizontal_flip': True,
        })
    return ImageDataGenerator(**kwargs)

def predict(model, data, steps=None, threshold=0.5):
    predictions = model.predict(data, steps=steps, verbose=1)
    return predictions, np.where(predictions >= threshold, 1, 0)

def temp(test_data_dir, batch_size, shuffle=False):
    test_datagen = get_datagen(use_default_augmentation=False)
    return test_datagen.flow_from_directory(
        directory=test_data_dir,
        target_size=(IMG_WIDTH, IMG_WIDTH),
        batch_size=batch_size,
        class_mode=None,
        shuffle=shuffle
    )

def main():
    
    model_exp = load_model('run_Model2_best_model.keras')
    
    data = temp('vid/',64)
    predictions = model_exp.predict(data)
    print(predictions)
    if predictions.mean() > 0.5:
        print('Real')
    else:
        print('Fake')
      
    return 0

if __name__ == "__main__":
    main()