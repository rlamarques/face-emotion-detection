from keras import backend as K
from keras.applications.mobilenetv2 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import pickle

from sklearn.preprocessing import MinMaxScaler

import numpy as np


def load_model_from_file(model_path):
    keras_backend = K.backend()
    assert keras_backend == "tensorflow", \
        "Only tensorflow-backed Keras models are supported, tried to load Keras model " \
        "with backend %s." % (keras_backend)
    return load_model(model_path)

def get_feature_vector(data, model, learning_phase=0):
    """ Returns the second-to-last layer output from a pretrained model

    Params
    ------
    data: ndarray. Data to input into the model, must match its shape.
    model: keras.engine.training.Model. Pretrained model
    learning_phase: int. If the model has a different behavior in
        training/testing phase, a suitable `learning_phase` must be 
        set: 0=TEST (default), 1=TRAIN.
    Return
    ------
    ndarray. The feature array for all the images.
    """

    get_layer_output = K.function(
        [model.layers[0].input, K.learning_phase()], 
        [model.layers[-2].output])
      
    return get_layer_output([data, learning_phase])[0]

if __name__ == '__main__':
    IMAGE_SHAPE = (224, 224, 3)
    BATCH_SIZE=100

    total_valid_images = 13616
    steps = (total_valid_images//BATCH_SIZE)+1

    IMAGE_DIR = {
        'test':  './organized/test'
    }

    datagen_attrs = dict(
        batch_size=BATCH_SIZE,  # How many images will be used in each step
        target_size=IMAGE_SHAPE[:2],  # Resize to fit models' input
        class_mode='categorical'  # Return labels as 1D integer label array
    )

    max_array = np.fromfile('max.ndarray')
    min_array = np.fromfile('min.ndarray')

    # Feature Normalization 
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit([min_array, max_array])

    datagen = lambda: ImageDataGenerator(preprocessing_function=preprocess_input)
    test_it = datagen().flow_from_directory(IMAGE_DIR['test'], **datagen_attrs)
    mnv2_classify_emotions  = load_model_from_file('weights-improvement-128x128-226-0.73.hdf5')

    with open('sgd-epoch-5-(valscore-0.606).pkl','rb') as fid:
        clf = pickle.load(fid) 

    ground_truth_list = []
    best_predictions = []
    with open('predictions.txt','w') as f
        for i in range(steps):
            X, Y = next(test_it)
            features = get_feature_vector(X, mnv2_classify_emotions)  
            features = min_max_scaler.transform(features)
            predictions = clf.predict(features)
            for j, y in enumerate(Y):
                ground_truth = np.argmax(y)
                ground_truth_list.append(ground_truth)

                best_predictions.append(predictions[j])
                f.write("%d %d %d\n" % (i*BATCH_SIZE + j, ground_truth, predictions[j]))
            print('Step', i)

