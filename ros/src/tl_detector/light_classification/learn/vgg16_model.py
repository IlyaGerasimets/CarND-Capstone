from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Flatten, Dense
from keras.models import Model

import numpy as np

nb_classes = 3

def get_model():
    # load model without last layer (expected input 224x224x3)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

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


model = get_model()

img_path = 'images/img_0_0.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
print(features)
