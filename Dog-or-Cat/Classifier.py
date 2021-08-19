from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('./train',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

model.fit(training_set,
                         steps_per_epoch = 25000//32,
                         epochs = 25,
                         validation_steps = 2000)

model_json = model.to_json()
with open("./model.json","w") as json_file:
  json_file.write(model_json)


model.save_weights("./model.h5")

print("Classifier trained Successfully!")