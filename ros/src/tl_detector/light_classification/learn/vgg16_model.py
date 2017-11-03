from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Flatten, Dense
from keras.models import Model, load_model
from keras.preprocessing.image import img_to_array

import tensorflow as tf
import os
import cv2

import numpy as np

nb_classes = 3
input_shape = (800, 600, 3)
source_shape = (160, 120, 3) # original VGG expects input 224x224x3
batch_size = 64
nb_epoch = 25

def get_model():
    # load model without last layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=source_shape)

    # new model with 3 classes
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return model

def train_model(model):
    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    datagen = image.ImageDataGenerator(width_shift_range=.2, height_shift_range=.2, shear_range=0.05, zoom_range=.1, 
        fill_mode='nearest', rescale=1. / 255)

    image_data_gen = datagen.flow_from_directory('images', target_size=(source_shape[0], source_shape[1]), 
        classes=['green', 'yellow', 'red'], batch_size=batch_size)

    model.fit_generator(image_data_gen, steps_per_epoch=50, epochs=nb_epoch)


def save_model_state(model):
    filename = 'nets/light_classifier_model_vgg16_%sx%s.h5'% (source_shape[0], source_shape[1])
    model.save(filename)

def load_result_model():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model = load_model(dir_path + '/nets/light_classifier_model_vgg16_160x120_25.h5')
    #graph = tf.get_default_graph()
    model._make_predict_function()
    return model

def im_debug(img):
    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.imshow('dst_rt', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#model = get_model()
#model.summary()
#train_model(model)
#save_model_state(model)
def test_image_convert(model, image_path):
    cv_image = cv2.imread(image_path)
    # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    cv_image = cv2.resize(cv_image, (160, 120)) # .astype(np.float32)
    image_data = np.reshape(cv_image, (1,160,120,3))
    x = preprocess_input(image_data)
    features = model.predict(x)
    print(features) # GYR

model = load_result_model()
test_image_convert(model, 'images/red/img_0_0.png')
test_image_convert(model, 'images/yellow/img_97_4.png')
test_image_convert(model, 'images/green/img_174_4.png')
