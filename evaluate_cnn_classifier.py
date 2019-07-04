from keras import backend as K
from keras.applications.mobilenetv2 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import numpy as np


def load_model_from_file(model_path):
    keras_backend = K.backend()
    assert keras_backend == "tensorflow", \
        "Only tensorflow-backed Keras models are supported, tried to load Keras model " \
        "with backend %s." % (keras_backend)
    return load_model(model_path)


if __name__ == '__main__':
    IMAGE_SHAPE = (224, 224, 3)
    BATCH_SIZE = 100

    total_valid_images = 13616
    steps = (total_valid_images // BATCH_SIZE) + 1

    IMAGE_DIR = {
        'test':  './organized/test'
    }

    datagen_attrs = dict(
        batch_size=BATCH_SIZE,  # How many images will be used in each step
        target_size=IMAGE_SHAPE[:2],  # Resize to fit models' input
        class_mode='categorical'  # Return labels as 1D integer label array
    )

    datagen = lambda: ImageDataGenerator(preprocessing_function=preprocess_input)
    test_it = datagen().flow_from_directory(IMAGE_DIR['test'], **datagen_attrs)
    mnv2_classify_emotions = load_model_from_file('weights-improvement-128x128-226-0.73.hdf5')

    ground_truth_list = []
    best_predictions = []
    with open('predictions.txt','w') as f:
        for i in range(steps):
            X, Y = next(test_it)
            predictions = mnv2_classify_emotions.predict(X)
            for j, y in enumerate(Y):
                ground_truth = np.argmax(y)
                ground_truth_list.append(ground_truth)
                # Prediction array
                sorted_predictions = np.argsort(predictions[j])
                best_predictions.append(sorted_predictions[-1])
                # Save predictions to file
                top_1 = sorted_predictions[-1]
                top_2 = sorted_predictions[-2]
                f.write("%d %d %d %d\n" % (i*BATCH_SIZE + j, ground_truth, top_1, top_2))
            print('Step', i)