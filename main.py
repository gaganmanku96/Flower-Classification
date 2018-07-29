from keras.layers import MaxPooling2D, Dense, Flatten, Convolution2D
from keras.models import Sequential


model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode ="same", input_shape=(128, 128,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode ="same", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, border_mode ="same", activation='relu'))


model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dense(5,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('flowers/training',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('flowers/test',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model.fit_generator(training_set,
                         steps_per_epoch = 3636,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 1122)



