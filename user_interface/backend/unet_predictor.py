import os
import cv2
import numpy as np

from unet.UNet import UNet
from unet.utils import pipeline

def _generate_model():
    current_dir = os.getcwd()
    params = {
        'loss': 'sparse_categorical_crossentropy',
        'batch_size': 16,
        'optimizer': 'sgd',
        'epochs': 2,
        'image_shape': (224, 224),
        'seed': 47,
        'apply_augmentation': True,
        'augmentation_threshold': 0.4,
        'checkpoint_path': '.'
    }
    model = UNet(params)
    model.restore_weights()
    return model


def get_prediction(image_file_location):
    image = cv2.imread(image_file_location)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    model = _generate_model()

    prediction = model.predict(image_file_location)
    masked_image = pipeline.get_masked_image(image, prediction)
    
    return masked_image