from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Flatten, Dense
from keras.models import Model

import numpy as np

nb_classes = 3
input_shape = (800, 600, 3)
source_shape = (160, 120, 3) # original VGG expects input 224x224x3
batch_size = 64
nb_epoch = 10

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
    datagen = image.ImageDataGenerator(width_shift_range=.2, height_shift_range=.2, shear_range=0.05, zoom_range=.1, 
        fill_mode='nearest', rescale=1. / 255)

    image_data_gen = datagen.flow_from_directory('images', target_size=(source_shape[0], source_shape[1]), 
        classes=['green', 'yellow', 'red'], batch_size=batch_size)

    model.fit_generator(image_data_gen, steps_per_epoch=50)


def save_model_state(model):
    filename = 'nets/light_classifier_model_vgg16_%sx%s.h5'% (source_shape[0], source_shape[1])
    model.save(filename)


model = get_model()
model.summary()
train_model(model)
save_model_state(model)

# img_path = 'images/img_0_0.png'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# features = model.predict(x)
# print(features)
