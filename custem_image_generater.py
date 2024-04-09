import os
import json, cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from prepare_image import prepare_data, show_polygon
from random import shuffle

def image_generator(directory, batch_size=32):
    images = []
    for file in os.listdir(directory):
        if "images" == file:
            images_dir = os.path.join(directory, file)
            for file in os.listdir(images_dir):
                if not ".png" in file:
                    continue
                images.append(os.path.join(images_dir, file))
                if len(images) == batch_size:
                    yield images
                    images = []
    yield images


class CustomDataGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        pass

    def flow_from_directory_with_annotations(self, directory, input_shape, batch_size=32):
        while True:
            images = next(image_generator(directory, batch_size))
            annotations = []
            images_new = []
            shuffle(images)
            for i, image_path in enumerate(images):
                label_path = image_path.replace('images', 'labels')
                annotation_path = os.path.splitext(label_path)[0] + '.json'
                image = cv2.imread(image_path)

                with open(annotation_path) as f:
                    annotation = json.load(f)['points']
                    image, annotation = prepare_data(image, input_shape, annotation)
                    annotations.append(annotation)
                    images_new.append(image)
                    #show_polygon(image, annotation)

            annotations = np.array(annotations)
            images_new = np.array(images_new)
            yield images_new, annotations

