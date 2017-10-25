from keras.preprocessing.image import ImageDataGenerator
import model

batch_size = 64
nb_epoch = 25

model = model.get_model()
model.summary()
datagen = ImageDataGenerator(width_shift_range=.2, height_shift_range=.2, shear_range=0.05, zoom_range=.1,
                             fill_mode='nearest', rescale=1. / 255)
image_data_gen = datagen.flow_from_directory('images', target_size=(800, 600), classes=['red', 'yellow', 'green'],
                                             batch_size=batch_size)
model.fit_generator(image_data_gen, nb_epoch=nb_epoch, samples_per_epoch=image_data_gen.nb_sample, verbose=1)

model.save('light_classifier_model.h5')
