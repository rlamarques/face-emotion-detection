from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import time

IMAGES_SHAPE=(150, 150, 3)
# BASE_DIR = Path("/content/drive/My Drive")
IMAGE_DIR =  './organized/training'
VALIDATION_IMAGE_DIR =  './organized/validation'
# TEST_IMAGE_DIR = BASE_DIR / 'datasett/organized/test'

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

images_dimensions=(224, 224)
batch_size=20
epochs=400
class_mode='categorical'

train_it = train_datagen.flow_from_directory(IMAGE_DIR, batch_size=batch_size, target_size=images_dimensions)
val_it = valid_datagen.flow_from_directory(VALIDATION_IMAGE_DIR, batch_size=batch_size, target_size=images_dimensions)
# test_it = datagen.flow_from_directory(TEST_IMAGE_DIR, batch_size=batch_size, target_size=images_dimensions, class_mode=class_mode)

num_classes=len(train_it.class_indices)
print(train_it.class_indices)

base_mnv2 = MobileNetV2(input_shape=images_dimensions + tuple([3]), include_top=False, weights='imagenet', pooling='avg')

custom_output = Dense(num_classes, activation='softmax', name='predictions')(base_mnv2.output)
custom_mnv2 = Model(inputs=base_mnv2.input, outputs=custom_output)

custom_mnv2.summary()

filepath= "./weights/weights-improvement-128x128-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
tensorboard = TensorBoard(log_dir= "./board/{}".format(time.time()))
callbacks_list = [tensorboard, checkpoint]

adadelta = Adadelta(lr=0.5, rho=0.95, epsilon=1e-6)
custom_mnv2.compile(loss='mean_squared_error', optimizer=adadelta, metrics=['accuracy'])

custom_mnv2.fit_generator(train_it, epochs=epochs, steps_per_epoch=batch_size//2, validation_data=val_it, validation_steps=batch_size//4, callbacks=callbacks_list)

start=time.clock()
# custom_mnv2.evaluate_generator(test_it, steps=len(test_it), verbose=1)
print('Elapsed time:', time.clock() - start)