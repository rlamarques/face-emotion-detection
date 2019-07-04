import pickle
from pathlib import Path

import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler

from keras import backend as K
from keras.applications.mobilenetv2 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf


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


def go(model, clf, classes, batch_size=100, max_iter=1000, tol=1e-5, epochs=300, steps_per_epoch=10, validation_steps=5):
    scores = {}
    val_scores = {}
    
    last_iter_score = 0
    for epoch in range(1, epochs+1):
        scores[epoch] = {}
        # For each batch
        for epoch_step in range(1, steps_per_epoch + 1):
            X, Y = next(train_it)
            # Get feature vector from CNN and normalize it
            features = get_feature_vector(X, model)
            features = min_max_scaler.transform(features)

            # Iter in this batch until max_iter or enhancement < tol
            for i in range(max_iter):
                clf.partial_fit(features, Y, classes=classes)
                iter_score = clf.score(features, Y)

                # Check iter enhancement
                if iter_score - last_iter_score < tol:
                    break

                last_iter_score = iter_score

            print('Epoch %d (%d/%d) - score: %.5f' % (epoch, epoch_step, steps_per_epoch, iter_score), end='\r')
            scores[epoch][epoch_step] = iter_score

        # Reprint the last epoch step 
        print('Epoch %d (%d/%d) - score: %.5f' % (epoch, epoch_step, steps_per_epoch, iter_score), end='')

        # Run some validation steps to compute the score
        epoch_val_scores = []
        for val_step in range(validation_steps):
            X, Y = next(val_it)
            # Get feature vector from CNN and normalize it
            features = get_feature_vector(X, model)
            features = min_max_scaler.transform(features)
            val_score = clf.score(features, Y)
            
            epoch_val_scores.append(val_score)

        val_scores[epoch] = np.average(epoch_val_scores)
        print(' - val_score: %.2f' % (val_scores[epoch]))
        
        # save the classifier
        with open('sgd-epoch-%d-(valscore-%.3f).pkl' % (epoch, val_scores[epoch]), 'wb') as fid:
            pickle.dump(clf, fid)    

        yield


if __name__=='__main__':

    # Setup GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True    #avoid getting all available memory in GPU
    config.gpu_options.visible_device_list = "0"  #set which GPU to use
    sess = tf.Session(config=config)

    # Feature Normalization 
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit([min_array, max_array])
        
    IMAGE_SHAPE = (224, 224, 3)
    BATCH_SIZE=500
    BASE_DIR = Path(".")
    IMAGE_DIR = {
        'train':  './organized/training',
        'validation':   './organized/validation',
        'test':  './organized/test'
    }
    # Prepare to load the images
    datagen_attrs = dict(
        batch_size=BATCH_SIZE,  # How many images will be used in each step
        target_size=IMAGE_SHAPE[:2],  # Resize to fit models' input
        class_mode='sparse'  # Return labels as 1D integer label array
    )

    datagen = lambda: ImageDataGenerator(preprocessing_function=preprocess_input)

    print('Setting up train data:', end=' ')
    train_it = datagen().flow_from_directory(IMAGE_DIR['train'], **datagen_attrs)
    print('Setting up validation data:', end=' ')
    val_it = datagen().flow_from_directory(IMAGE_DIR['validation'], **datagen_attrs)

    num_classes = len(val_it.class_indices)

    # Print classes summary
    print('Loaded', num_classes, 'classes:', val_it.class_indices)

    clf = SGDClassifier(
        penalty='l2',
        loss='hinge',
        random_state=0,
        tol=1e-3,
        n_jobs=8
    )
    mnv2_classify_emotions = load_model_from_file('weights-improvement-128x128-226-0.73.hdf5')

    for _ in go(mnv2_classify_emotions, clf, classes=np.arange(7), batch_size=BATCH_SIZE,
                steps_per_epoch=BATCH_SIZE//5, validation_steps=BATCH_SIZE//6):
        pass

